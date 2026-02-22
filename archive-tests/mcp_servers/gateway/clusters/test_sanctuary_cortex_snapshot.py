
import pytest
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.mcp_servers.gateway.gateway_test_client import GatewayTestClient

def test_cortex_snapshot_tool_registered():
    """Verify that cortex_capture_snapshot is registered and discoverable via the Gateway."""
    client = GatewayTestClient()
    
    # 1. Health check
    if not client.health_check():
        pytest.fail("Gateway is not reachable. Ensure 'mcp_servers/gateway' is running (Port 4444).")

    # 2. Discovery on 'cortex' scope (or 'sanctuary_cortex' alias if mapped)
    # The slug usually maps to the server name in fleet_registry.json
    # Based on fleet_setup.py, the name is likely 'cortex' or 'sanctuary_cortex'.
    # We'll try listing all tools or specifically query the 'cortex' server.
    
    # Let's list known servers first to be safe, or just list all tools.
    # The GatewayTestClient.list_tools(slug) calls get_mcp_tools(slug).
    
    # Try fetching tools from 'cortex' server (standard alias)
    try:
        response = client.list_tools(gateway_slug="cortex")
        tools = response.get("tools", [])
        tool_names = [t["name"] for t in tools]
        
        # NOTE: Gateway prefixes tool names with the server slug/alias.
        # Check against the actual namespaced tool name.
        target_tool = "sanctuary-cortex-cortex-capture-snapshot"
        
        assert target_tool in tool_names, \
            f"Tool '{target_tool}' not found in cortex tools. Available: {tool_names}"
            
    except Exception as e:
        # Fallback: Check if it's registered under 'sanctuary_cortex'
        try:
            response = client.list_tools(gateway_slug="sanctuary_cortex")
            tools = response.get("tools", [])
            tool_names = [t["name"] for t in tools]
            
            target_tool = "sanctuary-cortex-cortex-capture-snapshot"
            assert target_tool in tool_names, \
                f"Tool '{target_tool}' not found in sanctuary_cortex tools. Available: {tool_names}"
        except Exception as inner_e:
             pytest.fail(f"Could not discover tools from cortex or sanctuary_cortex: {e} / {inner_e}")

def test_cortex_snapshot_schema():
    """Verify the schema of the cortex_capture_snapshot tool."""
    client = GatewayTestClient()
    response = client.list_tools(gateway_slug="sanctuary_cortex")
    tools = response.get("tools", [])
    
    target_tool = "sanctuary-cortex-cortex-capture-snapshot"
    snapshot_tool = next((t for t in tools if t["name"] == target_tool), None)
    assert snapshot_tool is not None, f"Tool {target_tool} not found"
    
    # Check schema properties
    schema = snapshot_tool.get("inputSchema", {})
    props = schema.get("properties", {})
    
    assert "manifest_files" in props
    assert "snapshot_type" in props
    assert "strategic_context" in props
