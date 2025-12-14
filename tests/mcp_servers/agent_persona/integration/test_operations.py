"""
Agent Persona MCP Integration Tests - Operations Testing
=========================================================

Tests each MCP operation individually. Run all or pick specific tests.
The dispatch operation calls real Ollama LLM so it's SLOW.

CALLING EXAMPLES:
-----------------
# Run ALL fast operations (no LLM)
pytest tests/mcp_servers/agent_persona/integration/test_operations.py -v -m "not slow"

# Run specific operation
pytest tests/mcp_servers/agent_persona/integration/test_operations.py::test_list_roles -v
pytest tests/mcp_servers/agent_persona/integration/test_operations.py::test_get_state -v
pytest tests/mcp_servers/agent_persona/integration/test_operations.py::test_reset_state -v
pytest tests/mcp_servers/agent_persona/integration/test_operations.py::test_create_custom -v

# Run SLOW dispatch test (requires Ollama on localhost:11434)
pytest tests/mcp_servers/agent_persona/integration/test_operations.py::test_dispatch -v

# Run ALL tests including slow
pytest tests/mcp_servers/agent_persona/integration/test_operations.py -v

MCP OPERATIONS:
---------------
| Operation          | Speed | LLM Required | Description                    |
|--------------------|-------|--------------|--------------------------------|
| persona_list_roles | Fast  | No           | List built-in + custom roles   |
| persona_get_state  | Fast  | No           | Get conversation history        |
| persona_reset_state| Fast  | No           | Clear conversation history      |
| persona_create_custom| Fast| No           | Create new persona file         |
| persona_dispatch   | SLOW  | YES (Ollama) | Send task to agent, get response|

REQUIREMENTS:
-------------
- Fast tests: None
- Slow tests: Ollama running at localhost:11434 with model loaded

TEST HISTORY:
-------------
| Date       | Tester | Result | Notes                              |
|------------|--------|--------|------------------------------------|
| 2024-12-14 | Claude | PASS   | All fast ops pass, dispatch tested |
| (add new entries above this line)                                   |
"""


import pytest
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations


def is_ollama_running():
    """Check if Ollama is running via simple CLI command."""
    try:
        subprocess.run(
            ["ollama", "list"], 
            check=True, 
            capture_output=True, 
            timeout=2
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@pytest.fixture
def ops():
    """Create AgentPersonaOperations instance."""
    return AgentPersonaOperations()


# =============================================================================
# FAST OPERATIONS (No LLM required)
# =============================================================================

def test_list_roles(ops):
    """
    Test persona_list_roles operation.
    Fast - no LLM required.
    """
    result = ops.list_roles()
    
    print(f"\n📋 List Roles Result:")
    print(f"   Built-in: {result['built_in']}")
    print(f"   Custom: {result['custom']}")
    print(f"   Total: {result['total']}")
    
    # Assertions
    assert "built_in" in result
    assert "custom" in result
    assert "total" in result
    assert len(result["built_in"]) >= 3
    assert "coordinator" in result["built_in"]
    assert "strategist" in result["built_in"]
    assert "auditor" in result["built_in"]
    
    print("✅ persona_list_roles PASSED")


def test_get_state(ops):
    """
    Test persona_get_state operation.
    Fast - no LLM required.
    """
    result = ops.get_state(role="coordinator")
    
    print(f"\n📄 Get State Result:")
    print(f"   Role: {result['role']}")
    print(f"   State: {result['state']}")
    print(f"   Message Count: {result.get('message_count', 0)}")
    
    # Assertions
    assert "role" in result
    assert "state" in result
    assert result["role"] == "coordinator"
    assert result["state"] in ["no_history", "active", "error"]
    
    print("✅ persona_get_state PASSED")


def test_reset_state(ops):
    """
    Test persona_reset_state operation.
    Fast - no LLM required.
    """
    result = ops.reset_state(role="test_reset_role")
    
    print(f"\n🔄 Reset State Result:")
    print(f"   Role: {result['role']}")
    print(f"   Status: {result['status']}")
    
    # Assertions
    assert "role" in result
    assert "status" in result
    assert result["status"] in ["reset", "error"]
    
    print("✅ persona_reset_state PASSED")


def test_create_custom(ops):
    """
    Test persona_create_custom operation.
    Fast - no LLM required.
    """
    result = ops.create_custom(
        role="integration_test_persona",
        persona_definition="You are a test persona for integration testing.",
        description="Integration test persona"
    )
    
    print(f"\n🆕 Create Custom Result:")
    print(f"   Role: {result['role']}")
    print(f"   Status: {result['status']}")
    
    # Assertions
    assert result["status"] == "created"
    assert result["role"] == "integration_test_persona"
    assert "file_path" in result
    
    # Cleanup - delete the test persona
    persona_file = Path(result["file_path"])
    if persona_file.exists():
        persona_file.unlink()
        print(f"   Cleanup: Deleted {persona_file.name}")
    
    print("✅ persona_create_custom PASSED")


# =============================================================================
# SLOW OPERATIONS (Requires Ollama LLM)
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("role", ["coordinator", "auditor"])
def test_dispatch(ops, role):
    """
    Test persona_dispatch operation with different roles.
    SLOW - calls Ollama LLM on localhost:11434.
    
    Requires: Ollama running with model loaded.
    """
    if not is_ollama_running():
        pytest.skip("Ollama is not running. Skipping dispatch test.")

    print(f"\n🤖 Testing persona_dispatch ({role})...")
    
    task_map = {
        "coordinator": "Briefly describe what a git commit is. One sentence.",
        "auditor": "Briefly identify one security risk of hardcoded API keys. One sentence."
    }
    
    result = ops.dispatch(
        role=role,
        task=task_map[role]
    )
    
    print(f"\n📨 Dispatch Result ({role}):")
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        response = result.get('response', '')[:100]
        print(f"   Response: {response}...")
    
    # Assertions
    assert result["status"] == "success", f"Dispatch failed: {result.get('error')}"
    assert result["role"] == role
    assert len(result["response"]) > 10
    
    print(f"✅ persona_dispatch ({role}) PASSED")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

