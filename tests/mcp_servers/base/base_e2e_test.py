import pytest
import os
from pathlib import Path
from tests.mcp_servers.base.mcp_test_client import MCPTestClient

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class BaseE2ETest:
    """
    Base class for E2E tests that need to spin up an MCP server process.
    """
    
    SERVER_MODULE = None # e.g., "mcp_servers.chronicle.server"
    SERVER_NAME = None   # e.g., "chronicle"
    
    @pytest.fixture(scope="class", autouse=True)
    def mcp_client(self):
        """
        Fixture that starts the MCP server for the class and yields a client.
        Prioritizes SERVER_MODULE if set, otherwise falls back to SERVER_NAME path.
        """
        if self.SERVER_MODULE:
            client = MCPTestClient(self.SERVER_MODULE, is_module=True)
        elif self.SERVER_NAME:
            # Resolve server path: mcp_servers/<name>/server.py
            server_path = PROJECT_ROOT / "mcp_servers" / self.SERVER_NAME / "server.py"
            if not server_path.exists():
                raise FileNotFoundError(f"Server not found at {server_path}")
            client = MCPTestClient(server_path)
        else:
            yield None
            return

        # Setup environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        # Add any E2E specific flags (e.g. use test database?)
        
        client.start(env=env)
        try:
            yield client
        finally:
            client.stop()
    def call_tool(self, mcp_client, tool_name, args={}):
        """Helper to call tool with assertions."""
        result = mcp_client.call_tool(tool_name, args)
        if hasattr(result, "content"):
             # FastMCP response structure usually has 'content' or 'toolResult'
             pass
        return result
