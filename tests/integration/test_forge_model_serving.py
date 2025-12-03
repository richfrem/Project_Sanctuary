import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.system.forge.operations import ForgeOperations

@pytest.mark.integration
def test_forge_model_serving():
    """Test Forge serving model status and inference."""
    
    # Mock ollama module functions directly
    with patch('ollama.list') as mock_list, \
         patch('ollama.chat') as mock_chat:
        
        # Setup mock response for list
        mock_list.return_value = {
            "models": [
                {"name": "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"},
                {"name": "llama3:latest"}
            ]
        }
        
        # Setup mock response for chat
        mock_chat.return_value = {
            "message": {
                "content": "This is a response from the Sanctuary model."
            },
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        
        ops = ForgeOperations(project_root=".")
        
        # 1. Check model availability
        status = ops.check_model_availability()
        assert status["status"] == "success"
        assert status["available"] is True
        
        # 2. Query model
        response = ops.query_sanctuary_model(
            prompt="Explain Protocol 101",
            system_prompt="You are an expert."
        )
        
        assert response.response == "This is a response from the Sanctuary model."
        assert response.model == ops.sanctuary_model
        assert response.status == "success"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        
        # Verify ollama was called correctly
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert call_args[1]["model"] == ops.sanctuary_model
        assert call_args[1]["messages"][0]["content"] == "You are an expert."
