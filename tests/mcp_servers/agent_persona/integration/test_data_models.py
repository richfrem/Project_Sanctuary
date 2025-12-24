"""
Integration Tests for Agent Persona Pydantic Models.
Verifies Front Door schema validation.
"""
import pytest
from pydantic import ValidationError
from mcp_servers.agent_persona.models import (
    PersonaDispatchParams,
    PersonaRoleParams,
    PersonaCreateCustomParams
)

class TestPersonaSchemaValidation:

    def test_dispatch_valid(self):
        """Verify valid dispatch params."""
        req = PersonaDispatchParams(
            role="coordinator",
            task="Plan the mission",
            context={"project": "Alpha"},
            engine="openai"
        )
        assert req.role == "coordinator"
        assert req.maintain_state is True # Default
        assert req.context["project"] == "Alpha"

    def test_dispatch_missing_role(self):
        """Verify missing role raises error."""
        with pytest.raises(ValidationError) as excinfo:
            PersonaDispatchParams(task="Do something")
        assert "role" in str(excinfo.value)

    def test_role_params(self):
        """Verify simple role params."""
        req = PersonaRoleParams(role="auditor")
        assert req.role == "auditor"

    def test_create_custom_valid(self):
        """Verify custom persona creation params."""
        req = PersonaCreateCustomParams(
            role="coder",
            persona_definition="You are a coding bot",
            description="Writes code"
        )
        assert req.role == "coder"
        assert req.description == "Writes code"

    def test_create_custom_missing_definition(self):
        """Verify missing definition raises error."""
        with pytest.raises(ValidationError):
            PersonaCreateCustomParams(
                role="coder",
                description="Writes code"
            )
