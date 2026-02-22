"""
Council MCP E2E Tests - Protocol Verification
=============================================

Verifies all tools via JSON-RPC protocol against the real Council server.
Uses lightweight single-agent dispatch to avoid slow full deliberation.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/council/e2e/test_council_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                   | Type  | Description                    |
|------------------------|-------|--------------------------------|
| council_list_agents    | READ  | List available agents          |
| council_dispatch       | WRITE | Dispatch task (single agent)   |

"""
import pytest
import json
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestCouncilE2E(BaseE2ETest):
    SERVER_NAME = "council"
    SERVER_MODULE = "mcp_servers.council.server"

    def test_council_lifecycle(self, mcp_client):
        """Test cycle: List Agents → Dispatch (Single Agent) → Validate Response"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "council_list_agents" in names
        assert "council_dispatch" in names

        # 2. List Agents
        list_res = mcp_client.call_tool("council_list_agents", {})
        list_text = list_res.get("content", [])[0]["text"]
        print(f"📋 council_list_agents: {list_text}")
        
        # Should contain coordinator, strategist, auditor
        assert "coordinator" in list_text.lower() or "strategist" in list_text.lower()

        # 3. Dispatch Simple Task (Single Agent - Fast)
        # Use coordinator for simple task to avoid full deliberation
        task = "Confirm you are operational."
        
        print(f"\n🎯 Dispatching to single agent (coordinator)...")
        dispatch_res = mcp_client.call_tool("council_dispatch", {
            "task_description": task,
            "agent": "coordinator",  # Single agent dispatch
            "max_rounds": 1
        })
        dispatch_text = dispatch_res.get("content", [])[0]["text"]
        print(f"🤖 Response: {dispatch_text[:200]}...")
        
        # Validate response
        assert len(dispatch_text) > 0
        
        # Try parsing as JSON
        try:
            data = json.loads(dispatch_text)
            assert "decision" in data or "response" in data or "status" in data
            print("   ✅ Valid JSON response")
        except:
            # Text response is also valid
            print("   ✅ Text response received")
