import asyncio
import io
import sys
import unittest
import os
from unittest.mock import MagicMock, patch
from mcp_servers.lib.fastmcp_stub import FastMCP

class TestFastMCPModes(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_mode(self):
        """Verify run_stdio processes input from stdin and writes to stdout."""
        mcp = FastMCP("test-server")
        
        # Mock stdin with a JSON-RPC initialize request
        # FastMCP uses the underlying mcp-python-sdk which expects a specific sequence
        # For the stub, we implemented a simple loop.
        mock_stdin = io.StringIO('{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}\n')
        mock_stdout = io.StringIO()
        
        # Patch sys.stdin and sys.stdout
        with patch('sys.stdin', mock_stdin), \
             patch('sys.stdout', mock_stdout):
            
            # Run the stdio loop
            # We use a timeout or a way to break the loop if needed, 
            # but our stub's run_stdio breaks when it gets an empty line.
            await mcp.run_stdio()
            
            output = mock_stdout.getvalue()
            # print(f"Captured Output: {output}")
            
            self.assertIn('"result"', output)
            self.assertIn('"id": 1', output)

    @patch('uvicorn.run')
    def test_run_sse_mode(self, mock_uvicorn):
        """Verify run(transport='sse', port=...) calls uvicorn."""
        mcp = FastMCP("test-server")
        mcp.run(transport="sse", port=9999)
        mock_uvicorn.assert_called_once()
        args, kwargs = mock_uvicorn.call_args
        self.assertEqual(kwargs['port'], 9999)

    @patch('mcp_servers.lib.fastmcp_stub.FastMCP.run_stdio')
    def test_default_transport_stdio(self, mock_run_stdio):
        """Verify transport='stdio' triggers run_stdio."""
        mcp = FastMCP("test-server")
        # Since run() calls asyncio.run(self.run_stdio()), we need to mock that interaction
        # or just check where it goes.
        with patch('asyncio.run') as mock_asyncio_run:
            mcp.run(transport="stdio")
            mock_asyncio_run.assert_called()

if __name__ == '__main__':
    unittest.main()
