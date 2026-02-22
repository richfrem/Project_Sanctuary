import pytest
from unittest.mock import MagicMock, patch
import sys
from mcp_servers.forge_llm.operations import ForgeOperations
from mcp_servers.forge_llm.models import ModelQueryResponse

class TestForgeOperationsUnit:
    """Unit tests for ForgeOperations (mocked)."""

    @pytest.fixture
    def ops(self):
        return ForgeOperations(project_root="/tmp/test_project")

    def test_init(self, ops):
        """Test initialization sets correct model."""
        assert ops.project_root == "/tmp/test_project"
        assert "Sanctuary-Qwen2-7B" in ops.sanctuary_model

    def test_query_sanctuary_model_success(self, ops):
        """Test successful model query."""
        # Setup mock
        mock_ollama = MagicMock()
        mock_response = {
            'message': {'content': 'Test Answer'},
            'prompt_eval_count': 10,
            'eval_count': 20
        }
        mock_ollama.chat.return_value = mock_response

        # Patch sys.modules to catch the local import inside the method
        with patch.dict(sys.modules, {'ollama': mock_ollama}):
            result = ops.query_sanctuary_model("Test Prompt")

        # Verify
        assert result.status == "success"
        assert result.response == "Test Answer"
        assert result.total_tokens == 30
        
        # Verify call arguments
        mock_ollama.chat.assert_called_once()
        call_args = mock_ollama.chat.call_args
        assert call_args.kwargs['model'] == ops.sanctuary_model
        assert call_args.kwargs['messages'][0]['content'] == "Test Prompt"

    def test_check_model_availability_success(self, ops):
        """Test model availability check."""
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {
            'models': [
                {'name': 'other-model'},
                {'name': ops.sanctuary_model}
            ]
        }

        with patch.dict(sys.modules, {'ollama': mock_ollama}):
            result = ops.check_model_availability()

        assert result['status'] == "success"
        assert result['available'] is True
        assert ops.sanctuary_model in result['all_models']

    def test_check_model_availability_missing(self, ops):
        """Test model availability when model is missing."""
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {'models': [{'name': 'other-model'}]}

        with patch.dict(sys.modules, {'ollama': mock_ollama}):
            result = ops.check_model_availability()

        assert result['status'] == "success"
        assert result['available'] is False
