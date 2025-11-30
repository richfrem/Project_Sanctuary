"""
Tests for Git Workflow MCP tool safety checks.
Verifies that high-risk operations are blocked or handled correctly.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path to allow importing server
sys.path.append(os.getcwd())

class TestGitToolSafety(unittest.TestCase):
    
    def setUp(self):
        # Patch the git_ops object in the server module
        self.patcher = patch('mcp_servers.system.git_workflow.server.git_ops')
        self.mock_git_ops = self.patcher.start()
        
        # Import the tools after patching
        from mcp_servers.system.git_workflow.server import (
            git_add, 
            git_start_feature,
            git_smart_commit,
            git_push_feature,
            git_finish_feature
        )
        # Access the underlying function from the FunctionTool object
        self.git_add = git_add.fn
        self.git_start_feature = git_start_feature.fn
        self.git_smart_commit = git_smart_commit.fn
        self.git_push_feature = git_push_feature.fn
        self.git_finish_feature = git_finish_feature.fn

    def tearDown(self):
        self.patcher.stop()

    def test_git_add_blocks_main(self):
        """Test that git_add blocks staging on main branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "main",
            "feature_branches": []
        }
        
        result = self.git_add(["test.txt"])
        
        self.assertIn("ERROR", result)
        self.assertIn("Cannot stage files on main branch", result)
        self.mock_git_ops.add.assert_not_called()

    def test_git_add_blocks_non_feature(self):
        """Test that git_add blocks staging on non-feature branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "develop",
            "feature_branches": []
        }
        
        result = self.git_add(["test.txt"])
        
        self.assertIn("ERROR", result)
        self.assertIn("must be on a feature branch", result)
        self.mock_git_ops.add.assert_not_called()

    def test_git_add_allows_feature(self):
        """Test that git_add allows staging on feature branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "feature/task-123-test",
            "feature_branches": ["feature/task-123-test"]
        }
        
        result = self.git_add(["test.txt"])
        
        self.assertIn("Staged 1 file(s)", result)
        self.mock_git_ops.add.assert_called_with(["test.txt"])

    def test_start_feature_idempotent_same_branch(self):
        """Test start_feature is idempotent when already on the branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "feature/task-123-test",
            "feature_branches": ["feature/task-123-test"],
            "local_branches": [{"name": "feature/task-123-test"}],
            "is_clean": True
        }
        
        result = self.git_start_feature("123", "test")
        
        self.assertIn("Already on feature branch", result)
        self.mock_git_ops.create_branch.assert_not_called()

    def test_start_feature_idempotent_switch(self):
        """Test start_feature switches to existing branch if not current."""
        self.mock_git_ops.status.return_value = {
            "branch": "main",
            "feature_branches": ["feature/task-123-test"],
            "local_branches": [{"name": "main"}, {"name": "feature/task-123-test"}],
            "is_clean": True
        }
        
        result = self.git_start_feature("123", "test")
        
        self.assertIn("Switched to existing feature branch", result)
        self.mock_git_ops.create_branch.assert_not_called()
        self.mock_git_ops.checkout.assert_called_with("feature/task-123-test")

    def test_start_feature_blocks_multiple(self):
        """Test start_feature blocks if ANOTHER feature branch exists."""
        self.mock_git_ops.status.return_value = {
            "branch": "main",
            "feature_branches": ["feature/task-999-other"],
            "local_branches": [{"name": "main"}, {"name": "feature/task-999-other"}],
            "is_clean": True
        }
        
        result = self.git_start_feature("123", "test")
        
        self.assertIn("ERROR", result)
        self.assertIn("Existing feature branch(es) detected", result)
        self.mock_git_ops.create_branch.assert_not_called()

    def test_start_feature_blocks_dirty(self):
        """Test start_feature blocks if working directory is dirty (for new branch)."""
        self.mock_git_ops.status.return_value = {
            "branch": "main",
            "feature_branches": [],
            "local_branches": [{"name": "main"}],
            "is_clean": False,
            "staged": ["file.txt"],
            "modified": [],
            "untracked": []
        }
        
        result = self.git_start_feature("123", "test")
        
        self.assertIn("ERROR", result)
        self.assertIn("Working directory has uncommitted changes", result)
        self.mock_git_ops.create_branch.assert_not_called()

    def test_smart_commit_blocks_main(self):
        """Test that git_smart_commit blocks committing on main branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "main"
        }
        
        result = self.git_smart_commit("test commit")
        
        self.assertIn("ERROR", result)
        self.assertIn("Cannot commit directly to main branch", result)
        self.mock_git_ops.commit.assert_not_called()

    def test_push_feature_blocks_main(self):
        """Test that git_push_feature blocks pushing main branch."""
        self.mock_git_ops.get_current_branch.return_value = "main"
        
        result = self.git_push_feature()
        
        self.assertIn("ERROR", result)
        self.assertIn("Cannot push main branch directly", result)
        self.mock_git_ops.push.assert_not_called()

    def test_finish_feature_blocks_main(self):
        """Test that git_finish_feature blocks finishing 'main' branch."""
        result = self.git_finish_feature("main")
        
        self.assertIn("ERROR", result)
        self.assertIn("Cannot finish 'main' branch", result)
        self.mock_git_ops.checkout.assert_not_called()

    def test_finish_feature_blocks_invalid_name(self):
        """Test that git_finish_feature blocks invalid branch names."""
        result = self.git_finish_feature("develop")
        
        self.assertIn("ERROR", result)
        self.assertIn("Invalid branch name", result)
        self.mock_git_ops.checkout.assert_not_called()

    def test_finish_feature_blocks_unmerged(self):
        """Test that git_finish_feature blocks if branch is not merged."""
        # Mock is_branch_merged to return False
        self.mock_git_ops.is_branch_merged.return_value = False
        
        result = self.git_finish_feature("feature/task-123-test")
        
        self.assertIn("ERROR", result)
        self.assertIn("NOT merged into main", result)
        self.mock_git_ops.delete_local_branch.assert_not_called()

    def test_smart_commit_blocks_non_feature(self):
        """Test that git_smart_commit blocks committing on non-feature branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "develop"
        }
        
        result = self.git_smart_commit("test commit")
        
        self.assertIn("ERROR", result)
        self.assertIn("must be on a feature branch", result)
        self.mock_git_ops.commit.assert_not_called()

    def test_smart_commit_blocks_no_staged_files(self):
        """Test that git_smart_commit blocks if no files are staged."""
        self.mock_git_ops.status.return_value = {
            "branch": "feature/task-123-test"
        }
        self.mock_git_ops.get_staged_files.return_value = []
        
        result = self.git_smart_commit("test commit")
        
        self.assertIn("ERROR", result)
        self.assertIn("No files staged for commit", result)
        self.mock_git_ops.commit.assert_not_called()

    def test_push_feature_blocks_non_feature(self):
        """Test that git_push_feature blocks pushing non-feature branch."""
        self.mock_git_ops.get_current_branch.return_value = "develop"
        
        result = self.git_push_feature()
        
        self.assertIn("ERROR", result)
        self.assertIn("must be on a feature branch", result)
        self.mock_git_ops.push.assert_not_called()

    def test_finish_feature_blocks_dirty_state(self):
        """Test that git_finish_feature blocks if working directory is dirty."""
        self.mock_git_ops.verify_clean_state.side_effect = RuntimeError("Working directory is not clean")
        
        result = self.git_finish_feature("feature/task-123-test")
        
        self.assertIn("Failed to finish feature", result)
        self.mock_git_ops.delete_local_branch.assert_not_called()

    def test_smart_commit_success(self):
        """Test that git_smart_commit succeeds with staged files on feature branch."""
        self.mock_git_ops.status.return_value = {
            "branch": "feature/task-123-test"
        }
        self.mock_git_ops.get_staged_files.return_value = ["file1.py", "file2.py"]
        self.mock_git_ops.commit.return_value = "abc123def456"
        
        result = self.git_smart_commit("test commit message")
        
        self.assertIn("Commit successful", result)
        self.assertIn("abc123def456", result)
        self.mock_git_ops.commit.assert_called_with("test commit message")

    def test_push_feature_success(self):
        """Test that git_push_feature succeeds and verifies remote hash."""
        self.mock_git_ops.get_current_branch.return_value = "feature/task-123-test"
        self.mock_git_ops.push.return_value = "Push successful"
        self.mock_git_ops.get_commit_hash.side_effect = lambda ref: "abc123def456" if ref in ["HEAD", "origin/feature/task-123-test"] else "different"
        
        result = self.git_push_feature()
        
        self.assertIn("Verified push", result)
        self.assertIn("abc123de", result)  # First 8 chars of hash
        self.assertIn("Create PR", result)
        self.mock_git_ops.push.assert_called_with("origin", "feature/task-123-test", force=False, no_verify=False)

    def test_push_feature_hash_mismatch_warning(self):
        """Test that git_push_feature warns when remote hash doesn't match local."""
        self.mock_git_ops.get_current_branch.return_value = "feature/task-123-test"
        self.mock_git_ops.push.return_value = "Push successful"
        # Simulate hash mismatch
        def mock_hash(ref):
            if ref == "HEAD":
                return "abc123def456"
            elif ref == "origin/feature/task-123-test":
                return "different789"
            return "other"
        self.mock_git_ops.get_commit_hash.side_effect = mock_hash
        
        result = self.git_push_feature()
        
        self.assertIn("WARNING", result)
        self.assertIn("does not match", result)
        self.assertIn("abc123de", result)  # Local hash
        self.assertIn("differen", result)  # Remote hash (first 8 chars)

    def test_finish_feature_success_merged(self):
        """Test that git_finish_feature succeeds if branch is merged."""
        # Mock is_branch_merged to return True
        self.mock_git_ops.is_branch_merged.return_value = True
        
        result = self.git_finish_feature("feature/task-123-test")
        
        self.assertIn("Finished feature", result)
        self.assertIn("Verified merge", result)
        self.mock_git_ops.delete_local_branch.assert_called_with("feature/task-123-test", force=True)

if __name__ == "__main__":
    unittest.main()
