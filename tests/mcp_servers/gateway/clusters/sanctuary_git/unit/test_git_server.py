"""
Unit tests for Sanctuary Git Cluster (Aggregator).
Smoke tests to verify module import and server initialization.

Updated 2024-12-24: Aligned with ADR-066 v1.3 dual-transport pattern
(SSEServer for Gateway, FastMCP for STDIO)
"""
import pytest
import sys
from unittest.mock import MagicMock, patch

class TestSanctuaryGit:
    def test_server_import(self):
        """Verify server module can be imported without errors."""
        with patch.dict(sys.modules, {
            "mcp_servers.git.operations": MagicMock(),
            "git": MagicMock()
        }):
            from mcp_servers.gateway.clusters.sanctuary_git import server
            # Verify required components exist
            assert hasattr(server, 'run_server'), "Missing run_server entry point"
            assert hasattr(server, 'run_sse_server'), "Missing SSE transport function"
            assert hasattr(server, 'run_stdio_server'), "Missing STDIO transport function"
            assert hasattr(server, 'get_ops'), "Missing get_ops function"
    
    def test_schema_definitions(self):
        """Verify tool schemas are defined correctly."""
        with patch.dict(sys.modules, {
            "mcp_servers.git.operations": MagicMock(),
            "git": MagicMock()
        }):
            from mcp_servers.gateway.clusters.sanctuary_git import server
            # Check schema definitions exist
            assert hasattr(server, 'SMART_COMMIT_SCHEMA')
            assert hasattr(server, 'ADD_SCHEMA')
            assert hasattr(server, 'DIFF_SCHEMA')
            assert hasattr(server, 'LOG_SCHEMA')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
