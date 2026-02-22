import pytest
import sys
import os
from pathlib import Path
from tests.mcp_servers.base.mcp_test_client import MCPTestClient

@pytest.mark.e2e
def test_adr_connectivity():
    """
    Simple connectivity test for ADR MCP server.
    Verifies that MCPTestClient can start a server, initialize it, and call a tool.
    """
    server_module = "mcp_servers.adr.server"
    
    # Setup environment
    env = os.environ.copy()
    project_root = os.getcwd() # Assuming running from project root
    env["PYTHONPATH"] = project_root
    env["PROJECT_ROOT"] = project_root
    
    print(f"Starting ADR server: {server_module}")
    client = MCPTestClient(server_module, is_module=True)
    
    try:
        client.start(env=env)
        
        # Test tool listing
        tools = client.list_tools()
        print(f"Tools found: {[t['name'] for t in tools]}")
        assert len(tools) > 0
        assert "adr_list" in [t['name'] for t in tools]
        
        # Test tool call
        result = client.call_tool("adr_list", {})
        print(f"ADR List result type: {type(result)}")
        # Result should be a list (FastMCP returns the function result directly in 'result' key, which we extract)
        # However, checking the raw response structure in client might be needed if it fails.
        # But our client returns response.get("result", {}) -> which might be the tool output string or json.
        # Let's see what happens.
        
    finally:
        client.stop()
