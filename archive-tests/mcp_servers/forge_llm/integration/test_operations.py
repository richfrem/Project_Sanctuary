"""
Forge LLM MCP Integration Tests - Operations Testing
=====================================================

Tests Forge LLM MCP operations against a real Ollama instance.
Requires Ollama running locally with the specific Sanctuary model or fallback.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/forge_llm/integration/test_operations.py -v -s

MCP OPERATIONS:
---------------
| Operation                    | Type  | Description                          |
|------------------------------|-------|--------------------------------------|
| check_sanctuary_model_status | READ  | Check if model is loaded/available   |
| query_sanctuary_model        | READ  | Query the model                      |
"""
import pytest
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.forge_llm.operations import ForgeOperations


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
    """Create ForgeOperations instance."""
    return ForgeOperations(str(project_root))


@pytest.fixture(autouse=True)
def check_prerequisites():
    """Skip tests if Ollama is not running."""
    if not is_ollama_running():
        pytest.skip("Ollama is not running or not installed.")


def test_check_sanctuary_model_status(ops):
    """Test check_sanctuary_model_status operation."""
    result = ops.check_model_availability()
    
    print(f"\n🧠 check_sanctuary_model_status:")
    print(f"   Status: {result['status']}")
    print(f"   Available: {result.get('available')}")
    print(f"   Model: {result.get('model')}")
    
    assert result['status'] == 'success'
    assert 'all_models' in result
    print("✅ PASSED")


def test_query_sanctuary_model(ops):
    """Test query_sanctuary_model operation."""
    # First check availability to conditionally skip
    status = ops.check_model_availability()
    if not status.get('available'):
        # Just warn if specific model missing, but don't fail suite if we can't test query
        # Ideally we fallback or skip
        pytest.skip(f"Sanctuary model {ops.sanctuary_model} not found in Ollama.")
    
    prompt = "Explain Project Sanctuary in 10 words."
    print(f"\n🤖 query_sanctuary_model:")
    print(f"   Prompt: {prompt}")
    
    result = ops.query_sanctuary_model(
        prompt=prompt,
        max_tokens=20,
        temperature=0.1
    )
    
    print(f"   Response: {result.response}")
    
    assert result.status == 'success'
    assert len(result.response) > 0
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
