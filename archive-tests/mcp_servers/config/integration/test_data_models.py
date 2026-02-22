"""
Integration Tests for Config Server Pydantic Models.
Verifies schema validation for configuration management.
"""
import pytest
from pydantic import ValidationError
from mcp_servers.config.models import (
    ConfigReadRequest,
    ConfigWriteRequest,
    ConfigDeleteRequest
)

class TestConfigSchemaValidation:

    def test_read_request_valid(self):
        """Verify valid read request."""
        req = ConfigReadRequest(filename="mcp_servers.json")
        assert req.filename == "mcp_servers.json"

    def test_read_request_missing_filename(self):
        """Verify missing filename raises error."""
        with pytest.raises(ValidationError):
            ConfigReadRequest()

    def test_write_request_valid(self):
        """Verify valid write request."""
        req = ConfigWriteRequest(
            filename="my_config.json",
            content='{"key": "value"}'
        )
        assert req.filename == "my_config.json"
        assert req.content == '{"key": "value"}'

    def test_write_request_missing_content(self):
        """Verify missing content raises error."""
        with pytest.raises(ValidationError):
            ConfigWriteRequest(filename="empty.json")

    def test_delete_request_valid(self):
        """Verify valid delete request."""
        req = ConfigDeleteRequest(filename="old_config.json")
        assert req.filename == "old_config.json"
