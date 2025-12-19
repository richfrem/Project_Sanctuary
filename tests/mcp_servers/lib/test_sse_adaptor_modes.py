import asyncio
import io
import sys
import unittest
from unittest.mock import MagicMock, patch
from mcp_servers.lib.sse_adaptor import SSEServer

class TestSSEServerModes(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_mode(self):
        """Verify run_stdio processes input from stdin and writes to stdout."""
        server = SSEServer("test-server")
        
        # Mock stdin with a JSON-RPC initialize request
        mock_stdin = io.StringIO('{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}\n')
        mock_stdout = io.StringIO()
        
        # Patch sys.stdin and sys.stdout
        with patch('sys.stdin', mock_stdin), \
             patch('sys.stdout', mock_stdout), \
             patch('asyncio.to_thread', side_effect=lambda func, *args: func(*args)):
            
            # We need to mock asyncio.to_thread because it runs in a separate thread 
            # and might not see the mocked sys.stdin if not handled carefully, 
            # but for a simple read it might be okay. 
            # Actually, SSEServer uses asyncio.to_thread(stdin.readline)
            
            # Let's run the server (it should exit after reading one line because readline returns empty string next)
            await server.run_stdio()
            
            output = mock_stdout.getvalue()
            print(f"Captured Output: {output}")
            
            self.assertIn('"result":', output)
            self.assertIn('"protocolVersion":', output)

    @patch('uvicorn.run')
    def test_run_http_mode(self, mock_uvicorn):
        """Verify run(port=...) calls uvicorn."""
        server = SSEServer("test-server")
        server.run(port=9999)
        mock_uvicorn.assert_called_once()
        args, kwargs = mock_uvicorn.call_args
        self.assertEqual(kwargs['port'], 9999)

    @patch('mcp_servers.lib.sse_adaptor.SSEServer.run_stdio')
    def test_transport_env_var(self, mock_run_stdio):
        """Verify MCP_TRANSPORT=stdio triggers run_stdio."""
        server = SSEServer("test-server")
        with patch.dict('os.environ', {'MCP_TRANSPORT': 'stdio'}):
            server.run()
            mock_run_stdio.assert_called_once()  # Should be awaited in real life, but logic path is checked

if __name__ == '__main__':
    unittest.main()
