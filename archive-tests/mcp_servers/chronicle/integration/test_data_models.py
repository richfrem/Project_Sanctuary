"""
Integration Tests for Chronicle Pydantic Models (Schema Validation).
Verifies that the "Front Door" correctly validates and sanitizes inputs.
"""
import pytest
from pydantic import ValidationError
from mcp_servers.chronicle.models import (
    ChronicleCreateRequest,
    ChronicleUpdateRequest,
    ChronicleGetRequest,
    ChronicleListRequest,
    ChronicleSearchRequest,
    ChronicleStatus,
    ChronicleClassification
)

class TestChronicleSchemaValidation:
    
    def test_create_request_valid(self):
        """Verify a valid create request is accepted."""
        req = ChronicleCreateRequest(
            title="Deploying Gateway",
            content="# Deployment\n\nSuccessful deployment.",
            author="DevOps",
            status="published",
            classification="public"
        )
        assert req.title == "Deploying Gateway"
        assert req.status == "published"

    def test_create_request_missing_fields(self):
        """Verify missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            ChronicleCreateRequest(
                title="Only Title"
                # Missing content, author
            )
        errors = excinfo.value.errors()
        missing = [e['loc'][0] for e in errors]
        assert "content" in missing
        assert "author" in missing

    def test_create_request_defaults(self):
        """Verify default values are applied."""
        req = ChronicleCreateRequest(
            title="Defaults Test",
            content="Testing defaults",
            author="QA"
        )
        assert req.status == "draft"
        assert req.classification == "internal"
        assert req.date is None

    def test_update_request_valid(self):
        """Verify valid update request."""
        req = ChronicleUpdateRequest(
            entry_number=5,
            updates={"status": "deprecated"},
            reason="Superseded by v2"
        )
        assert req.entry_number == 5
        assert req.updates["status"] == "deprecated"

    def test_get_request_validation(self):
        """Verify type coercion."""
        req = ChronicleGetRequest(entry_number="10") 
        assert req.entry_number == 10

    def test_list_request_defaults(self):
        """Verify default limit."""
        req = ChronicleListRequest()
        assert req.limit == 10
