"""
Integration Tests for ADR Server Tool Handlers.
Verifies that the server functions correctly accept Pydantic models
and delegate to the (mocked) operations layer.
"""
import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.adr import server
from mcp_servers.adr.models import ADRCreateRequest

class TestADRToolIntegration:

    @pytest.fixture
    def mock_ops(self):
        with patch("mcp_servers.adr.server.adr_ops") as mock:
            yield mock

    def test_create_adr_integration(self, mock_ops):
        """Verify adr_create handler accepts model and calls ops."""
        # Setup mock return
        mock_ops.create_adr.return_value = {
            "adr_number": 1,
            "file_path": "/path/to/001-title.md"
        }

        # Create valid Pydantic request
        request = ADRCreateRequest(
            title="Integration Test",
            context="Testing integration",
            decision="Use mocks",
            consequences="Fast tests",
            status="proposed"
        )

        # Call the tool handler directly (accessing underlying function via .fn)
        result = server.adr_create.fn(request)

        # Assertions
        assert "Successfully created ADR 001" in result
        mock_ops.create_adr.assert_called_once_with(
            title="Integration Test",
            context="Testing integration",
            decision="Use mocks",
            consequences="Fast tests",
            date=None,
            status="proposed",
            author="AI Assistant",
            supersedes=None
        )
