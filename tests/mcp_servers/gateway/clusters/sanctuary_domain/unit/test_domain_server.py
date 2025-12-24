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
            assert server.mcp is not None
            assert server.mcp.name == "sanctuary_domain"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
