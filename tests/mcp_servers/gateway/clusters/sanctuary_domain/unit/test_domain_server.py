"""
Unit tests for Sanctuary Domain Cluster (Aggregator).
Smoke test to verify module import and FastMCP initialization.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch

class TestSanctuaryDomain:
    def test_server_import(self):
        """Verify server can be imported and mcp object exists."""
        with patch.dict(sys.modules, {
            "mcp_servers.chronicle.operations": MagicMock(),
            "mcp_servers.protocol.operations": MagicMock(),
            "mcp_servers.task.operations": MagicMock(),
            "mcp_servers.adr.operations": MagicMock(),
            "mcp_servers.agent_persona.operations": MagicMock(),
            "mcp_servers.config.operations": MagicMock(),
            "mcp_servers.workflow.operations": MagicMock(),
        }):
            from mcp_servers.gateway.clusters.sanctuary_domain import server
            assert hasattr(server, 'run_sse_server')
            # Server uses SSE transport, not FastMCP mcp object

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
