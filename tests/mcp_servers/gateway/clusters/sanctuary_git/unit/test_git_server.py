"""
Unit tests for Sanctuary Git Cluster (Aggregator).
Smoke test to verify module import and FastMCP initialization.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch

class TestSanctuaryGit:
    def test_server_import(self):
        """Verify server can be imported and mcp object exists."""
        with patch.dict(sys.modules, {
            "mcp_servers.git.operations": MagicMock(),
            "git": MagicMock()
        }):
            from mcp_servers.gateway.clusters.sanctuary_git import server
            assert server.mcp is not None
            assert server.mcp.name == "sanctuary_git"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
