"""
Unit tests for Sanctuary Cortex Cluster (Aggregator).
Smoke test to verify module import and FastMCP initialization.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch

class TestSanctuaryCortex:
    def test_server_import(self):
        """Verify server can be imported and SSE transport function exists."""
        # Mock dependencies that might be heavy or not present
        with patch.dict(sys.modules, {
            "mcp_servers.rag_cortex.operations": MagicMock(),
            "mcp_servers.forge_llm.operations": MagicMock(),
            "chromadb": MagicMock(),
            "ollama": MagicMock()
        }):
            from mcp_servers.gateway.clusters.sanctuary_cortex import server
            assert hasattr(server, 'run_sse_server'), "Server must have run_sse_server for Gateway mode"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
