"""
Unit tests for Sanctuary FileSystem Cluster (Aggregator).
Smoke test to verify module import and FastMCP initialization.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch

class TestSanctuaryFileSystem:
    def test_server_import(self):
        """Verify server can be imported and mcp object exists."""
        with patch.dict(sys.modules, {
            "mcp_servers.code.operations": MagicMock(),
        }):
            from mcp_servers.gateway.clusters.sanctuary_filesystem import server
            assert server.mcp is not None
            assert server.mcp.name == "sanctuary_filesystem"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
