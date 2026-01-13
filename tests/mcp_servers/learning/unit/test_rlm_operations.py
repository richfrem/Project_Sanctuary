import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from mcp_servers.learning.operations import LearningOperations

@pytest.fixture
def learning_ops(tmp_path):
    """Fixture to initialize LearningOperations with a temp root."""
    return LearningOperations(project_root=tmp_path)

@pytest.fixture
def mock_repo(tmp_path):
    """Creates a dummy repo structure for testing RLM."""
    (tmp_path / "01_PROTOCOLS").mkdir()
    (tmp_path / "01_PROTOCOLS/Protocol_Test.md").write_text("# Test Protocol\nThis is a test.")
    (tmp_path / "ADRs").mkdir()
    (tmp_path / "ADRs/001_Test.md").write_text("# Test ADR\nThis is a test.")
    return tmp_path

class TestRLMOperations:

    @patch("requests.post")
    def test_rlm_map_ollama_call(self, mock_post, learning_ops, mock_repo):
        """
        Scenario: _rlm_map iterates files and calls Ollama.
        Expected: 2 files found, 2 calls to requests.post.
        """
        # Mock Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a summary."}
        mock_post.return_value = mock_response

        # Execute
        roots = ["01_PROTOCOLS", "ADRs"]
        # Note: We must ensure learning_ops uses the mock_repo as root
        learning_ops.project_root = mock_repo
        
        results = learning_ops._rlm_map(roots)

        # Assertions
        assert len(results) == 2
        assert "01_PROTOCOLS/Protocol_Test.md" in results
        assert "ADRs/001_Test.md" in results
        assert results["01_PROTOCOLS/Protocol_Test.md"] == "This is a summary."
        
        # Verify call arguments (checking prompt construction)
        assert mock_post.call_count == 2
        args, kwargs = mock_post.call_args_list[0]
        assert "model" in kwargs["json"]
        assert "prompt" in kwargs["json"]
        assert "options" in kwargs["json"]

    @patch("requests.post")
    def test_rlm_map_timeout_handling(self, mock_post, learning_ops, mock_repo):
        """
        Scenario: Ollama times out.
        Expected: Graceful failure handling (log warning, return error string).
        """
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out")
        
        learning_ops.project_root = mock_repo
        results = learning_ops._rlm_map(["01_PROTOCOLS"])
        
        assert "01_PROTOCOLS/Protocol_Test.md" in results
        assert "[RLM Read Timeout]" in results["01_PROTOCOLS/Protocol_Test.md"]

    def test_rlm_reduce_formatting(self, learning_ops):
        """
        Scenario: _rlm_reduce formats the map into a Markdown Hologram.
        Expected: Headers, counts, and summaries present.
        """
        mock_map = {
            "01_PROTOCOLS/P1.md": "Protocol Summary",
            "ADRs/A1.md": "ADR Summary",
            "mcp_servers/server.py": "Code Summary"
        }
        
        hologram = learning_ops._rlm_reduce(mock_map)
        
        assert "# Cognitive Hologram (Protocol 132)" in hologram
        assert "Protocol Summary" in hologram
        assert "ADR Summary" in hologram
        assert "Code Summary" in hologram
        assert "**Process Metrics:**" not in hologram # Metrics added at synthesis level, not reduce level
