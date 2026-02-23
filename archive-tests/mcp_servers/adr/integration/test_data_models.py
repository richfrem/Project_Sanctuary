"""
Integration Tests for ADR Pydantic Models (Schema Validation).
Verifies that the "Front Door" correctly validates and sanitizes inputs.
"""
import pytest
from pydantic import ValidationError
from mcp_servers.adr.models import (
    ADRCreateRequest,
    ADRUpdateStatusRequest,
    ADRGetRequest,
    ADRListRequest,
    ADRSearchRequest
)

class TestADRSchemaValidation:
    
    def test_create_request_valid(self):
        """Verify a valid create request is accepted."""
        req = ADRCreateRequest(
            title="Use Pydantic",
            context="Need validation",
            decision="We will use Pydantic",
            consequences="Better safety",
            author="Tester",
            status="proposed"
        )
        assert req.title == "Use Pydantic"
        assert req.status == "proposed"

    def test_create_request_missing_fields(self):
        """Verify missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            ADRCreateRequest(
                title="Only Title"
                # Missing context, decision, consequences
            )
        errors = excinfo.value.errors()
        missing = [e['loc'][0] for e in errors]
        assert "context" in missing
        assert "decision" in missing
        assert "consequences" in missing

    def test_create_request_defaults(self):
        """Verify default values are applied."""
        req = ADRCreateRequest(
            title="Defaults Test",
            context="Testing defaults",
            decision="Check defaults",
            consequences="None"
        )
        assert req.author == "AI Assistant"
        assert req.status == "proposed"
        assert req.date is None # Should default to None in model, handled by logic

    def test_update_status_valid(self):
        """Verify valid update status request."""
        req = ADRUpdateStatusRequest(
            number=1,
            new_status="accepted",
            reason="Approved by council"
        )
        assert req.number == 1
        assert req.new_status == "accepted"

    def test_get_request_validation(self):
        """Verify type coercion (strict mode usually, but Pydantic might coerce string '1' to int 1)."""
        # Pydantic V2 often coerces simple types by default unless strict=True
        req = ADRGetRequest(number="123") 
        assert req.number == 123
        assert isinstance(req.number, int)

    def test_search_request_empty(self):
        """Verify empty query behavior (depending on model constraints)."""
        # If strict min_length wasn't set, empty string might be allowed. 
        # Checking implementation: query: str = Field(...)
        req = ADRSearchRequest(query="")
        assert req.query == ""
