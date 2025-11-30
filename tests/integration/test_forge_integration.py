"""
Integration tests for Forge MCP Server

Tests the Sanctuary model query and status operations.
Requires Ollama with Sanctuary-Qwen2-7B model installed.
"""
import pytest
import subprocess
import json
from pathlib import Path


def check_ollama_installed():
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_sanctuary_model():
    """Check if Sanctuary model is available in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Check for the Sanctuary model
            expected_model = "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"
            return expected_model in result.stdout or "Sanctuary-Qwen2" in result.stdout
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="module")
def verify_prerequisites():
    """Verify all prerequisites before running tests."""
    if not check_ollama_installed():
        pytest.skip("Ollama is not installed. Install from https://ollama.ai")
    
    if not check_sanctuary_model():
        pytest.skip(
            "Sanctuary model not found in Ollama. Install with:\n"
            "ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"
        )


class TestForgePrerequisites:
    """Test Forge MCP prerequisites."""
    
    def test_ollama_installed(self):
        """Verify Ollama is installed and accessible."""
        assert check_ollama_installed(), (
            "Ollama is not installed or not in PATH. "
            "Install from https://ollama.ai"
        )
    
    def test_sanctuary_model_available(self):
        """Verify Sanctuary model is available in Ollama."""
        assert check_sanctuary_model(), (
            "Sanctuary model not found in Ollama. Install with:\n"
            "ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"
        )
    
    def test_ollama_list_output(self):
        """Verify ollama list shows expected model information."""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, "Failed to run 'ollama list'"
        
        # Check output format
        lines = result.stdout.strip().split('\n')
        assert len(lines) >= 2, "Expected header and at least one model"
        
        # Verify header
        header = lines[0]
        assert "NAME" in header, "Expected NAME column in header"
        assert "ID" in header or "SIZE" in header, "Expected ID or SIZE column in header"


class TestForgeOperations:
    """Test Forge MCP operations."""
    
    def test_check_sanctuary_model_status(self, verify_prerequisites):
        """Test check_sanctuary_model_status operation."""
        # Import here to avoid import errors if server not available
        from mcp_servers.system.forge.operations import ForgeOperations
        
        project_root = Path(__file__).parent.parent.parent
        ops = ForgeOperations(str(project_root))
        
        response = ops.check_model_availability()
        
        assert response["status"] == "success", f"Expected success, got {response['status']}: {response.get('error')}"
        assert response["available"] is True, "Model should be available"
        assert response["model"] is not None, "Model name should be set"
    
    def test_query_sanctuary_model(self, verify_prerequisites):
        """Test query_sanctuary_model operation."""
        from mcp_servers.system.forge.operations import ForgeOperations
        
        project_root = Path(__file__).parent.parent.parent
        ops = ForgeOperations(str(project_root))
        
        # Simple test query
        test_prompt = "What is Project Sanctuary?"
        response = ops.query_sanctuary_model(
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        assert response.status == "success", f"Expected success, got {response.status}: {response.error}"
        assert response.response is not None, "Response should not be None"
        assert len(response.response) > 0, "Response should not be empty"
        assert response.model is not None, "Model name should be set"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
