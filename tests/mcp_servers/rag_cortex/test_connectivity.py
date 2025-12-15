import pytest
import sys
import os
from tests.mcp_servers.base.mcp_test_client import MCPTestClient

@pytest.mark.e2e
def test_rag_cortex_connectivity():
    """
    Connectivity test specifically for RAG Cortex MCP server.
    """
    server_module = "mcp_servers.rag_cortex.server"
    
    # Setup environment
    env = os.environ.copy()
    project_root = os.getcwd()
    env["PYTHONPATH"] = project_root
    env["PROJECT_ROOT"] = project_root
    env["SKIP_CONTAINER_CHECKS"] = "1" # Crucial!
    
    # Mock RAG env vars
    env["CHROMA_HOST"] = "localhost"
    env["CHROMA_PORT"] = "8000"
    
    print(f"Starting RAG Cortex server: {server_module}")
    client = MCPTestClient(server_module, is_module=True)
    
    try:
        client.start(env=env)
        
        # Test tool listing
        tools = client.list_tools()
        print(f"Tools found: {[t['name'] for t in tools]}")
        assert len(tools) > 0
        assert "cortex_query" in [t['name'] for t in tools]
        
    finally:
        client.stop()
