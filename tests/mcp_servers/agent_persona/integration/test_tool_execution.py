"""
Integration Tests for Agent Persona Tool Handlers.
Verifies server.py deleg logic via .fn() access.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from mcp_servers.agent_persona import server
from mcp_servers.agent_persona.models import PersonaDispatchParams

class TestPersonaToolIntegration:

    @pytest.fixture
    def mock_ops(self):
        with patch("mcp_servers.agent_persona.server.persona_ops") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_persona_dispatch_integration(self, mock_ops):
        """Verify dispatch tool handler calls ops."""
        # Setup mock return
        mock_ops.dispatch.return_value = {
            "role": "coordinator",
            "response": "I have a plan",
            "status": "success"
        }

        # Create request
        req = PersonaDispatchParams(
            role="coordinator",
            task="Make a plan",
            maintain_state=True
        )

        # Call underlying function
        result = await server.persona_dispatch.fn(req)
        
        # Verify JSON serialization and ops call
        data = json.loads(result)
        assert data["response"] == "I have a plan"
        
        mock_ops.dispatch.assert_called_once_with(
            role="coordinator",
            task="Make a plan",
            context=None,
            maintain_state=True,
            engine=None,
            model_name=None,
            custom_persona_file=None
        )
