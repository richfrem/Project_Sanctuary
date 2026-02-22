import pytest
import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture(scope="session")
def check_ollama_available():
    """Check if Ollama is available and reachable."""
    try:
        import ollama
        # Try a lightweight call
        try:
            ollama.list()
            return True
        except Exception:
            return False
    except ImportError:
        return False

@pytest.fixture(autouse=True)
def skip_integration_if_no_ollama(request, check_ollama_available):
    """Auto-skip integration tests if Ollama is not running."""
    if request.node.get_closest_marker("integration"):
        if not check_ollama_available:
            pytest.skip("Skipping integration test: Ollama not available")

@pytest.fixture
def mock_ollama():
    """Mock the ollama library for unit tests."""
    with pytest.MonkeyPatch.context() as mp:
        mock = MagicMock()
        mp.setattr("mcp_servers.forge_llm.operations.ollama", mock)
        yield mock
