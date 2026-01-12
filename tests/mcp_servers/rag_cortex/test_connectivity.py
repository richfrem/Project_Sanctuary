import pytest
import sys
import os
import json
from tests.mcp_servers.base.mcp_test_client import MCPTestClient
from mcp_servers.lib.path_utils import find_project_root

@pytest.mark.e2e
def test_rag_cortex_connectivity():
    """
    Connectivity test specifically for RAG Cortex MCP server.
    """
    server_module = "mcp_servers.rag_cortex.server"
    
    # Setup environment
    env = os.environ.copy()
    project_root = find_project_root()
    env["PYTHONPATH"] = project_root
    env["PROJECT_ROOT"] = project_root
    env["SKIP_CONTAINER_CHECKS"] = "1" # Crucial!
    
    # Use env_helper to get values from .env
    from mcp_servers.lib.env_helper import get_env_variable
    
    # RAG env vars from .env
    env["CHROMA_HOST"] = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
    env["CHROMA_PORT"] = get_env_variable("CHROMA_PORT", required=False) or "8110"
    
    print(f"Connecting to Chroma at {env['CHROMA_HOST']}:{env['CHROMA_PORT']}")
    
    print(f"Starting RAG Cortex server: {server_module}")
    client = MCPTestClient(server_module, is_module=True)
    
    try:
        client.start(env=env)
        
        # Test tool listing
        tools = client.list_tools()
        print(f"Tools found: {[t['name'] for t in tools]}")
        assert len(tools) > 0
        tool_names = [t['name'] for t in tools]
        assert "cortex_query" in tool_names
        # assert "cortex_learning_debrief" in tool_names (Moved to Learning MCP)

        # Test functional call
        print("Calling cortex_get_stats...")
        mcp_response = client.call_tool("cortex_get_stats", {})
        # MCP response format is {'content': [{'type': 'text', 'text': '...'}]}
        text_content = mcp_response['content'][0]['text']
        result = json.loads(text_content)
        print(f"Stats result: {result}")
        
        assert "total_documents" in result
        assert result["total_documents"] >= 1000
        assert result["health_status"] in ["healthy", "degraded"]
        
        if result.get("error"):
             pytest.fail(f"Stats call failed with error: {result.get('error')}")
        
    finally:
        client.stop()
