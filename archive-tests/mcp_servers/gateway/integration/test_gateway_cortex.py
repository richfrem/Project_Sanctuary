
import pytest
import requests
import os
import urllib3
import json
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest

# Load environment variables from .env file
load_dotenv()

# Suppress InsecureRequestWarning for self-signed certs (Localhost/Podman)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@pytest.mark.integration
@pytest.mark.gateway
class TestGatewayCortex(BaseIntegrationTest):
    """
    Verification suite for Cortex functionality via the Sanctuary Gateway.
    Focus: Protocol 128 (Learning Continuity) and Tool Discovery.
    """
    
    def get_required_services(self) -> List[Tuple[str, int, str]]:
        """Declare dependency on the external Gateway container."""
        # We also implicitly need sanctuary_cortex (8104), but Gateway (4444) is the entrypoint.
        return [("localhost", 4444, "Sanctuary Gateway")]

    @property
    def config(self):
        """Lazy load config."""
        return {
            "URL": os.getenv("MCP_GATEWAY_URL", "https://localhost:4444"),
            "API_TOKEN": os.getenv("MCPGATEWAY_BEARER_TOKEN", "")
        }

    def test_tool_discovery_protocol_128(self):
        """
        Verify that Protocol 128 tools are discoverable via the Gateway.
        Target: GET /tools
        Expectation: 'sanctuary-cortex-cortex-learning-debrief' is present in the list.
        """
        url = f"{self.config['URL']}/tools"
        headers = {"Authorization": f"Bearer {self.config['API_TOKEN']}"}
        
        print(f"\n[Discovery] Listing tools from: {url}")
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        assert response.status_code == 200, f"Discovery Failed: {response.text}"
        
        # Gateway returns a list or a dict {data: []} depending on endpoint
        data = response.json()
        tools = data if isinstance(data, list) else data.get("tools", [])
        tool_names = [t["name"] for t in tools]
        
        print(f"Tools Found (count={len(tool_names)})")
        
        # Use canonical naming
        assert "sanctuary-cortex-cortex-learning-debrief" in tool_names, \
            "CRITICAL: 'sanctuary-cortex-cortex-learning-debrief' NOT found in Gateway."
        
        assert "sanctuary-cortex-cortex-guardian-wakeup" in tool_names, \
            "CRITICAL: 'sanctuary-cortex-cortex-guardian-wakeup' NOT found in Gateway."

    def test_learning_debrief_execution(self):
        """
        Verify that we can execute 'cortex_learning_debrief' via the Gateway.
        Target: POST /rpc
        """
        url = f"{self.config['URL']}/rpc"
        headers = {
            "Authorization": f"Bearer {self.config['API_TOKEN']}",
            "Content-Type": "application/json"
        }
        # Use the RPC format used in gateway_client.py
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "sanctuary-cortex-cortex-learning-debrief",
                "arguments": {
                    "content": "# Gateway Integration Test\nValidating Protocol 128 routing."
                }
            },
            "id": 1
        }
        
        print(f"\n[Execution] Calling cortex_learning_debrief via Gateway RPC...")
        response = requests.post(url, headers=headers, json=payload, verify=False, timeout=10)
        
        assert response.status_code == 200, f"Execution Failed: {response.text}"
        result = response.json()
        print(f"Result: {result}")
        
        assert "error" not in result, f"RPC Error: {result.get('error')}"
        
        # Check success logic
        # Result structure: {"result": {"content": [{"type": "text", "text": "..."}]}}
        content_blocks = result.get("result", {}).get("content", [])
        text_content = content_blocks[0].get("text", "") if content_blocks else ""
        
        assert "status" in text_content and "success" in text_content, \
            f"Result did not indicate success: {text_content}"

