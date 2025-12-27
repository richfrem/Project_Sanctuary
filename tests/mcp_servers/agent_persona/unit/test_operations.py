"""
Unit tests for Agent Persona Operations.
Verifies business logic with mocked LLM interactions.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mcp_servers.agent_persona.operations import PersonaOperations

@pytest.fixture
def mock_root(tmp_path):
    """Create a mock project root with necessary structure."""
    # Setup structure expected by PersonaOperations:
    # project_root/mcp_servers/agent_persona/{personas,state}
    base = tmp_path / "mcp_servers" / "agent_persona"
    personas = base / "personas"
    personas.mkdir(parents=True)
    
    # Create dummy built-in persona files
    (personas / "coordinator.txt").write_text("You are the Coordinator.")
    (personas / "strategist.txt").write_text("You are the Strategist.")
    (personas / "auditor.txt").write_text("You are the Auditor.")
    
    return tmp_path

@pytest.fixture
def persona_ops(mock_root):
    """Initialize PersonaOperations with mock root."""
    return PersonaOperations(project_root=mock_root)

def test_initialization(persona_ops, mock_root):
    """Test directory creation."""
    assert persona_ops.persona_dir.exists()
    assert persona_ops.state_dir.exists()
    assert persona_ops.persona_dir == mock_root / "mcp_servers" / "agent_persona" / "personas"

def test_list_roles(persona_ops):
    """Test listing roles picks up files."""
    roles = persona_ops.list_roles()
    # Check built-ins are detected (based on PersonaConstants, and we created files)
    assert "coordinator" in roles["built_in"]
    assert "strategist" in roles["built_in"]

def test_create_custom_persona(persona_ops):
    """Test creating a custom persona file."""
    result = persona_ops.create_custom(
        role="tester",
        persona_definition="You are a tester.",
        description="A test persona."
    )
    
    assert result["status"] == "created"
    assert (persona_ops.persona_dir / "tester.txt").exists()
    assert (persona_ops.persona_dir / "tester.txt").read_text() == "You are a tester."

def test_dispatch_success(persona_ops):
    """Test dispatch flow with mocked agent creation."""
    with patch.object(persona_ops, '_create_agent') as mock_create:
        # Setup mock agent
        mock_agent = MagicMock()
        mock_agent.query.return_value = "Plan executed."
        mock_create.return_value = mock_agent
        
        result = persona_ops.dispatch(
            role="coordinator",
            task="Make a plan",
            maintain_state=True
        )
        
        # Verify
        assert result["status"] == "success"
        assert result["role"] == "coordinator"
        assert result["response"] == "Plan executed."
        assert result["state_preserved"] is True
        
        # Check calls
        mock_create.assert_called_once()
        mock_agent.query.assert_called()
        mock_agent.save_history.assert_called_once()

def test_dispatch_error_handling(persona_ops):
    """Test dispatch handles agent errors gracefully."""
    with patch.object(persona_ops, '_create_agent') as mock_create:
        mock_agent = MagicMock()
        mock_agent.query.side_effect = Exception("LLM connection failed")
        mock_create.return_value = mock_agent
        
        result = persona_ops.dispatch("coordinator", "Task")
        
        assert result["status"] == "error"
        assert "LLM connection failed" in result["error"]

def test_reset_state(persona_ops):
    """Test state file deletion."""
    role = "coordinator"
    state_file = persona_ops.state_dir / f"{role}_session.json"
    state_file.write_text("[]")
    
    result = persona_ops.reset_state(role)
    
    assert result["status"] == "reset"
    assert not state_file.exists()

def test_get_state_empty(persona_ops):
    """Test getting state for new session."""
    state = persona_ops.get_state("coordinator")
    assert state["state"] == "no_history"
    assert state["messages"] == []
