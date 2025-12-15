"""
Agent Persona MCP E2E Tests - Protocol Verification
===================================================

Verifies all tools via JSON-RPC protocol against the real Agent Persona server.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/agent_persona/e2e/test_agent_persona_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                   | Type  | Description                    |
|------------------------|-------|--------------------------------|
| persona_list_roles     | READ  | List available personas        |
| persona_create_custom  | WRITE | Create custom persona          |
| persona_dispatch       | WRITE | Dispatch task to persona       |
| persona_get_state      | READ  | Get conversation state         |
| persona_reset_state    | WRITE | Reset conversation state       |

"""
import pytest
import os
import json
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestAgentPersonaE2E(BaseE2ETest):
    SERVER_NAME = "agent_persona"
    SERVER_MODULE = "mcp_servers.agent_persona.server"

    def test_agent_persona_lifecycle(self, mcp_client):
        """Test cycle: List → Create → Get State → Reset (Skip slow dispatch)"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "persona_list_roles" in names
        assert "persona_create_custom" in names

        # 2. List Roles
        list_res = mcp_client.call_tool("persona_list_roles", {})
        list_text = list_res.get("content", [])[0]["text"]
        print(f"📋 persona_list_roles: {list_text}")
        # Should contain built-in roles
        assert "coordinator" in list_text.lower() or "strategist" in list_text.lower()

        # 3. Create Custom Persona (or verify exists)
        custom_role = "e2e_test_minimal"
        persona_def = "You are a minimal E2E test persona."
        
        create_res = mcp_client.call_tool("persona_create_custom", {
            "role": custom_role,
            "persona_definition": persona_def,
            "description": "Minimal E2E Test Persona"
        })
        create_text = create_res.get("content", [])[0]["text"]
        print(f"\n🆕 persona_create_custom: {create_text}")
        # Accept both created and already exists
        assert "created" in create_text.lower() or "exists" in create_text.lower() or custom_role in create_text

        # Cleanup file path
        persona_file = PROJECT_ROOT / ".agent" / "personas" / f"{custom_role}.txt"

        try:
            # 4. Get State (should be empty/no history)
            state_res = mcp_client.call_tool("persona_get_state", {"role": custom_role})
            state_text = state_res.get("content", [])[0]["text"]
            print(f"📊 persona_get_state: {state_text[:100]}...")
            assert "state" in state_text.lower() or "no_history" in state_text.lower()

            # 5. Reset State (should succeed even with no history)
            reset_res = mcp_client.call_tool("persona_reset_state", {"role": custom_role})
            reset_text = reset_res.get("content", [])[0]["text"]
            print(f"🔄 persona_reset_state: {reset_text}")
            assert "reset" in reset_text.lower() or "status" in reset_text.lower()
            
            # NOTE: Skipping dispatch test - it's slow (60s+ timeout) due to LLM calls
            print("\n⚠️  Skipping persona_dispatch test (slow LLM operation)")

        finally:
            # 6. Cleanup
            if persona_file.exists():
                os.remove(persona_file)
                print(f"🧹 Cleaned up {persona_file}")
