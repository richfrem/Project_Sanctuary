import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def persona_root(tmp_path):
    """Create a temporary directory for agent persona tests."""
    root = tmp_path / "persona_test_root"
    root.mkdir()
    
    # Create required subdirs matching AgentPersonaOperations expectations
    (root / "mcp_servers" / "agent_persona" / "personas").mkdir(parents=True)
    (root / ".agent" / "memory" / "persona_state").mkdir(parents=True)
    
    return root

@pytest.fixture
def mock_project_root(persona_root):
    """Return the temporary root as the project root."""
    return persona_root

@pytest.fixture
def check_ollama_available():
    """Check if Ollama is available (for skipping integration tests)."""
    try:
        import ollama
        try:
            ollama.list()
            return True
        except Exception:
            return False
    except ImportError:
        return False
