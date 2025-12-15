import pytest
import sys
import os
from tests.mcp_servers.base.mcp_test_client import MCPTestClient

@pytest.mark.e2e
def test_chronicle_connectivity():
    """
    Connectivity test for Chronicle MCP server.
    """
    server_module = "mcp_servers.chronicle.server"
    
    # Setup environment
    env = os.environ.copy()
    project_root = os.getcwd()
    env["PYTHONPATH"] = project_root
    env["PROJECT_ROOT"] = project_root
    
    print(f"Starting Chronicle server: {server_module}")
    client = MCPTestClient(server_module, is_module=True)
    
    try:
        client.start(env=env)
        
        # Test tool listing
        tools = client.list_tools()
        print(f"Tools found: {[t['name'] for t in tools]}")
        assert len(tools) > 0
        assert "chronicle_create_entry" in [t['name'] for t in tools]
        
    finally:
        client.stop()
