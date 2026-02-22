"""
Integration Tests for Config Server Tool Handlers.
Verifies that the server functions correctly accept Pydantic models
and delegate to the (mocked) operations layer.
"""
import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.config import server
from mcp_servers.config.models import ConfigWriteRequest

class TestConfigToolIntegration:

    @pytest.fixture
    def mock_ops(self):
        with patch("mcp_servers.config.server.config_ops", create=True) as mock:
            # Note: server.py imports ConfigOperations as ConfigOperations, 
            # then instantiates it as 'ops'. 
            # However, looking at server.py: 'ops = ConfigOperations(CONFIG_DIR)'
            # So we should patch 'mcp_servers.config.server.ops'
            yield mock

    @pytest.mark.asyncio
    async def test_config_write_integration(self):
        """Verify config_write handler accepts model and calls ops."""
        # Use patch on the specific artifact server.ops
        with patch("mcp_servers.config.server.ops") as mock_ops:
            mock_ops.write_config.return_value = "/path/to/config.json"
            
            # Create valid request
            req = ConfigWriteRequest(
                filename="test.json",
                content='{"foo": "bar"}'
            )

            # Call underlying function
            result = await server.config_write.fn(req)

            # Assertions
            assert "Successfully wrote config" in result
            mock_ops.write_config.assert_called_once()
            
            # Verify basic JSON handling logic in server.py
            # If filename ends in .json, it tries to parse.
            args, _ = mock_ops.write_config.call_args
            assert args[0] == "test.json"
            assert args[1] == {"foo": "bar"} # Should be dict if parsed
