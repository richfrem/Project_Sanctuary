import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.system.forge.operations import ForgeOperations

@pytest.mark.integration
def test_forge_model_serving():
    """Test Forge serving model status and inference."""
    
    # Mock Ollama client
    with patch('mcp_servers.system.forge.operations.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Setup mock response for list
        mock_client.list.return_value = {
            "models": [
                {"name": "Sanctuary-Qwen2.5-7B:latest"},
                {"name": "llama3:latest"}
            ]
        }
        
        # Setup mock response for chat
        mock_client.chat.return_value = {
            "message": {
                "content": "This is a response from the Sanctuary model."
            },
            "done": True
        }
        
        ops = ForgeOperations()
        
        # 1. Check model status
        status = ops.check_model_status()
        assert status["status"] == "available"
        assert status["model"] == "Sanctuary-Qwen2.5-7B:latest"
        
        # 2. Query model
        response = ops.query_model(
            prompt="Explain Protocol 101",
            system_prompt="You are an expert."
        )
        
        assert "response" in response
        assert response["response"] == "This is a response from the Sanctuary model."
        assert response["model"] == "Sanctuary-Qwen2.5-7B:latest"
        
        # Verify Ollama was called correctly
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]["model"] == "Sanctuary-Qwen2.5-7B:latest"
        assert call_args[1]["messages"][0]["content"] == "You are an expert."
