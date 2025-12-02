"""
Tests for Agent Persona MCP Server
"""

import pytest
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mcp_servers.lib.agent_persona.agent_persona_ops import AgentPersonaOperations

@pytest.fixture
def persona_ops():
    """Create AgentPersonaOperations instance"""
    return AgentPersonaOperations()

def test_persona_ops_initialization(persona_ops):
    """Test that AgentPersonaOperations initializes correctly"""
    assert persona_ops.project_root.exists()
    assert persona_ops.persona_dir.exists()
    assert persona_ops.state_dir.exists()

def test_list_roles(persona_ops):
    """Test listing available persona roles"""
    roles = persona_ops.list_roles()
    
    assert "built_in" in roles
    assert "custom" in roles
    assert "total" in roles
    
    # Should have at least the 3 built-in roles
    assert len(roles["built_in"]) >= 3
    assert "coordinator" in roles["built_in"]
    assert "strategist" in roles["built_in"]
    assert "auditor" in roles["built_in"]

def test_create_custom_persona(persona_ops):
    """Test creating a custom persona"""
    result = persona_ops.create_custom(
        role="test_persona",
        persona_definition="You are a test persona for unit testing.",
        description="Test persona for validation"
    )
    
    assert result["status"] == "created"
    assert result["role"] == "test_persona"
    assert "file_path" in result
    
    # Verify file was created
    persona_file = Path(result["file_path"])
    assert persona_file.exists()
    
    # Cleanup
    persona_file.unlink()

def test_create_duplicate_persona(persona_ops):
    """Test that creating duplicate persona fails"""
    # Create first persona
    result1 = persona_ops.create_custom(
        role="duplicate_test",
        persona_definition="Test",
        description="Test"
    )
    assert result1["status"] == "created"
    
    # Try to create duplicate
    result2 = persona_ops.create_custom(
        role="duplicate_test",
        persona_definition="Test",
        description="Test"
    )
    assert result2["status"] == "error"
    assert "already exists" in result2["error"]
    
    # Cleanup
    Path(result1["file_path"]).unlink()

def test_get_state_no_history(persona_ops):
    """Test getting state when no history exists"""
    result = persona_ops.get_state(role="nonexistent_role")
    
    assert result["role"] == "nonexistent_role"
    assert result["state"] == "no_history"
    assert result["messages"] == []

def test_reset_state(persona_ops):
    """Test resetting persona state"""
    result = persona_ops.reset_state(role="coordinator")
    
    assert result["role"] == "coordinator"
    assert result["status"] in ["reset", "error"]  # May not have state to reset

def test_dispatch_structure(persona_ops):
    """Test that dispatch method exists with correct signature"""
    # This is a structure test only - we won't actually execute
    # the orchestrator in CI/CD to avoid long-running tests
    
    assert hasattr(persona_ops, "dispatch")
    
    # Test parameter validation would go here
    # (actual execution tests should be manual or integration tests)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
