import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock FastMCP to allow direct function calls
mock_fastmcp = MagicMock()
mock_fastmcp.tool = lambda: lambda func: func
sys.modules["fastmcp"] = MagicMock()
sys.modules["fastmcp"].FastMCP = MagicMock(return_value=mock_fastmcp)

from mcp_servers.lib.git.git_ops import GitOperations

class TestPillar4Enforcement(unittest.TestCase):
    def setUp(self):
        self.git_ops = GitOperations()
        self.git_ops._run_git = MagicMock()

    def test_verify_clean_state_clean(self):
        """Test verify_clean_state with a clean repo."""
        # Mock status returning empty lists (clean)
        self.git_ops.status = MagicMock(return_value={
            "branch": "main",
            "staged": [],
            "modified": [],
            "untracked": []
        })
        
        try:
            self.git_ops.verify_clean_state()
        except RuntimeError:
            self.fail("verify_clean_state raised RuntimeError unexpectedly!")

    def test_verify_clean_state_dirty(self):
        """Test verify_clean_state with a dirty repo."""
        # Mock status returning modified files
        self.git_ops.status = MagicMock(return_value={
            "branch": "main",
            "staged": [],
            "modified": ["dirty_file.py"],
            "untracked": []
        })
        
        with self.assertRaises(RuntimeError) as cm:
            self.git_ops.verify_clean_state()
        
        self.assertIn("Working directory is not clean", str(cm.exception))

    @patch("mcp_servers.system.git_workflow.server.git_ops")
    def test_git_start_feature_enforcement(self, mock_git_ops):
        """Verify git_start_feature calls verify_clean_state."""
        from mcp_servers.system.git_workflow.server import git_start_feature
        
        # Mock verify_clean_state to succeed
        mock_git_ops.verify_clean_state = MagicMock()
        mock_git_ops.create_branch = MagicMock()
        mock_git_ops.checkout = MagicMock()
        
        git_start_feature("123", "test-feature")
        
        mock_git_ops.verify_clean_state.assert_called_once()

    @patch("mcp_servers.system.git_workflow.server.git_ops")
    def test_git_finish_feature_enforcement(self, mock_git_ops):
        """Verify git_finish_feature calls verify_clean_state."""
        from mcp_servers.system.git_workflow.server import git_finish_feature
        
        # Mock verify_clean_state to succeed
        mock_git_ops.verify_clean_state = MagicMock()
        mock_git_ops.checkout = MagicMock()
        mock_git_ops.pull = MagicMock()
        mock_git_ops.delete_branch = MagicMock()
        mock_git_ops.delete_remote_branch = MagicMock()
        
        git_finish_feature("feature/test")
        
        mock_git_ops.verify_clean_state.assert_called_once()

if __name__ == "__main__":
    unittest.main()
