"""
Integration Tests for Protocol Server Tool Handlers.
Verifies that the server functions correctly accept Pydantic models
and delegate to the (mocked) operations layer.
"""
import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.protocol import server
from mcp_servers.protocol.models import ProtocolCreateRequest

class TestProtocolToolIntegration:

    @pytest.fixture
    def mock_ops(self):
        with patch("mcp_servers.protocol.server.ops") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_protocol_create_integration(self, mock_ops):
        """Verify protocol_create handler accepts model and calls ops."""
        # Setup mock return
        mock_ops.create_protocol.return_value = {
            "protocol_number": 99,
            "file_path": "099_test.md"
        }

        # Create valid request
        req = ProtocolCreateRequest(
            number=99,
            title="Test Protocol",
            status="PROPOSED",
            classification="Internal",
            version="1.0",
            authority="TestAuth",
            content="# Content"
        )

        # Call underlying function
        result = await server.protocol_create.fn(req)

        # Assertions
        assert "Created Protocol 99" in result
        mock_ops.create_protocol.assert_called_once_with(
            99,
            "Test Protocol",
            "PROPOSED",
            "Internal",
            "1.0",
            "TestAuth",
            "# Content",
            None
        )
