import pytest
import os
from mcp_servers.forge_llm.operations import ForgeOperations

@pytest.mark.integration
class TestFeaturesIntegration:
    """Integration tests that require real Ollama."""

    @pytest.fixture
    def ops(self):
        return ForgeOperations(project_root=os.getcwd())

    def test_real_ollama_connection(self, ops):
        """Verify we can actually talk to Ollama."""
        result = ops.check_model_availability()
        
        # We don't assert it IS available (model might not be pulled),
        # but the CALL itself should succeed (status='success').
        assert result['status'] == 'success', f"Ollama connection failed: {result.get('error')}"
        assert 'all_models' in result

    def test_real_model_query(self, ops):
        """Attempt a real query if the model is loaded."""
        status = ops.check_model_availability()
        if not status.get('available'):
            pytest.skip(f"Sanctuary model {ops.sanctuary_model} not found in Ollama")

        # Use a very short generate request to save time/compute
        result = ops.query_sanctuary_model("Hi", max_tokens=5)
        
        assert result.status == 'success'
        assert len(result.response) > 0
