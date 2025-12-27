"""
Integration Tests for Protocol Server Pydantic Models.
Verifies schema validation for protocol management.
"""
import pytest
from pydantic import ValidationError
from mcp_servers.protocol.models import (
    ProtocolCreateRequest,
    ProtocolUpdateRequest,
    ProtocolGetRequest,
    ProtocolListRequest,
    ProtocolSearchRequest,
    ProtocolStatus
)

class TestProtocolSchemaValidation:

    def test_create_request_valid(self):
        """Verify valid create request."""
        req = ProtocolCreateRequest(
            number=99,
            title="Deploy Procedure",
            status="PROPOSED",
            classification="Internal",
            version="1.0",
            authority="LeadDev",
            content="# Deploy Content"
        )
        assert req.number == 99
        assert req.status == "PROPOSED"

    def test_create_request_missing_fields(self):
        """Verify missing fields raise validation error."""
        with pytest.raises(ValidationError):
            ProtocolCreateRequest(
                number=99,
                title="Partial Protocol"
                # Missing others
            )

    def test_update_request_valid(self):
        """Verify valid update request."""
        req = ProtocolUpdateRequest(
            number=99,
            updates={"status": "CANONICAL"},
            reason="Approved"
        )
        assert req.number == 99
        assert req.updates["status"] == "CANONICAL"

    def test_list_request_filter(self):
        """Verify optional status filter."""
        req = ProtocolListRequest(status="CANONICAL")
        assert req.status == "CANONICAL"
        
        req_none = ProtocolListRequest()
        assert req_none.status is None
