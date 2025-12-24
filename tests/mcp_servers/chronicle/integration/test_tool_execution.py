"""
Integration Tests for Chronicle Server Tool Handlers.
Verifies that the server functions correctly accept Pydantic models
and delegate to the (mocked) operations layer.
"""
import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.chronicle import server
from mcp_servers.chronicle.models import ChronicleCreateRequest

class TestChronicleToolIntegration:

    @pytest.fixture
    def mock_ops(self):
        with patch("mcp_servers.chronicle.server.ops") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_create_entry_integration(self, mock_ops):
        """Verify chronicle_create_entry handler accepts model and calls ops."""
        # Setup mock return
        mock_ops.create_entry.return_value = {
            "entry_number": 42,
            "file_path": "/path/to/042_deployment.md"
        }

        # Create valid Pydantic request
        request = ChronicleCreateRequest(
            title="Deployment",
            content="We deployed.",
            author="DevOps",
            status="published"
        )

        # Call the tool handler directly (accessing underlying function via .fn for FastMCP)
        result = await server.chronicle_create_entry.fn(request)

        # Assertions
        assert "Created Chronicle Entry 42" in result
        mock_ops.create_entry.assert_called_once_with(
            title="Deployment",
            content="We deployed.",
            author="DevOps",
            date=None,
            status="published",
            classification="internal" # Default value
        )
