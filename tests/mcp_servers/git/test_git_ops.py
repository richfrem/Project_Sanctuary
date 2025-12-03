import unittest
import os
import shutil
import tempfile
import subprocess
from mcp_servers.git.git_ops import GitOperations

class TestGitOperations(unittest.TestCase):
    """
    Test suite for GitOperations class (Protocol 101 v3.0 compliant).
    
    Note: Manifest generation tests have been removed as Protocol 101 v3.0
    uses Functional Coherence (test suite execution) instead of manifests.
    
    SAFETY RULES FOR GIT WORKFLOW MCP:
    1. Always check status first (git_get_status) before any operation
    2. One feature branch at a time - never create multiple concurrent branches
    3. Never commit directly to main - feature branches only
    4. git_finish_feature requires user confirmation that PR is merged
    5. git_sync_main should not be called while feature branch is active
    6. git_smart_commit automatically runs test suite (P101 v3.0)
    """
    
    def setUp(self):
        # Create a temporary directory for the repo
        self.test_dir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        
        # Create initial commit so HEAD exists
        with open("README.md", "w") as f:
            f.write("# Test Repo")
        subprocess.run(["git", "add", "README.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit", "--no-verify"], check=True)
        
        self.git_ops = GitOperations(self.test_dir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir)

    # PROTOCOL 101 v3.0: Manifest generation tests REMOVED
    # Functional Coherence (test suite execution) is now the integrity mechanism

    def test_commit_basic(self):
        """Test basic commit functionality (without manifest)."""
        # Create a file
        with open("test.txt", "w") as f:
            f.write("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        # Commit (using --no-verify to skip pre-commit hook in test environment)
        commit_hash = self.git_ops.commit("test commit")
        
        # Verify commit was created
        self.assertIsNotNone(commit_hash)
        self.assertEqual(len(commit_hash), 40)  # SHA-1 hash length

    def test_status(self):
        """Test repository status retrieval with enhanced branch info."""
        # Create a file
        with open("test.txt", "w") as f:
            f.write("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        status = self.git_ops.status()
        
        # Check basic fields
        self.assertEqual(status["branch"], "main")
        self.assertIn("test.txt", status["staged"])
        
        # Check enhanced fields
        self.assertIn("local_branches", status)
        self.assertIn("feature_branches", status)
        self.assertIn("remote", status)
        self.assertIn("is_clean", status)
        
        # Should have at least main branch
        self.assertGreaterEqual(len(status["local_branches"]), 1)
        
        # Should not be clean (has staged file)
        self.assertFalse(status["is_clean"])
        
        # No feature branches yet
        self.assertEqual(len(status["feature_branches"]), 0)

    def test_branch_operations(self):
        """Test branch creation, checkout, and deletion."""
        # Create branch
        self.git_ops.create_branch("feature/test")
        
        # Checkout
        self.git_ops.checkout("feature/test")
        self.assertEqual(self.git_ops.get_current_branch(), "feature/test")
        
        # Switch back
        self.git_ops.checkout("main")
        self.assertEqual(self.git_ops.get_current_branch(), "main")
        
        # Delete branch
        self.git_ops.delete_branch("feature/test")
        
        # Verify deletion (checkout should fail)
        with self.assertRaises(RuntimeError):
            self.git_ops.checkout("feature/test")

    def test_get_staged_files(self):
        """Test retrieval of staged files."""
        # Create and stage a file
        with open("test.txt", "w") as f:
            f.write("content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        staged = self.git_ops.get_staged_files()
        self.assertIn("test.txt", staged)

    def test_push_with_no_verify(self):
        """Test push with no_verify parameter (bypasses pre-push hooks)."""
        # Create a file and commit
        with open("test.txt", "w") as f:
            f.write("test content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        self.git_ops.commit("test commit for push")
        
        # Note: This test verifies the parameter is accepted and passed to git
        # In a real scenario with a remote, this would bypass pre-push hooks
        # For now, we just verify it doesn't raise an error
        try:
            # This will fail without a remote, but should fail gracefully
            self.git_ops.push(remote="origin", no_verify=True)
        except RuntimeError as e:
            # Expected to fail without remote, but should contain git error, not parameter error
            self.assertIn("fatal", str(e).lower())

    def test_push_with_force(self):
        """Test push with force parameter."""
        # Create a file and commit
        with open("test2.txt", "w") as f:
            f.write("test content 2")
        subprocess.run(["git", "add", "test2.txt"], check=True)
        self.git_ops.commit("test commit for force push")
        
        # Note: This test verifies the parameter is accepted and passed to git
        # In a real scenario with a remote, this would force push
        # For now, we just verify it doesn't raise an error
        try:
            # This will fail without a remote, but should fail gracefully
            self.git_ops.push(remote="origin", force=True)
        except RuntimeError as e:
            # Expected to fail without remote, but should contain git error, not parameter error
            self.assertIn("fatal", str(e).lower())

    def test_diff_unstaged(self):
        """Test diff for unstaged changes."""
        with open("test_diff.txt", "w") as f:
            f.write("original content")
        subprocess.run(["git", "add", "test_diff.txt"], check=True)
        self.git_ops.commit("add test_diff.txt")
        
        with open("test_diff.txt", "w") as f:
            f.write("modified content")
        
        diff_output = self.git_ops.diff(cached=False)
        self.assertIn("test_diff.txt", diff_output)

    def test_diff_staged(self):
        """Test diff for staged changes."""
        with open("test_staged.txt", "w") as f:
            f.write("staged content")
        subprocess.run(["git", "add", "test_staged.txt"], check=True)
        
        diff_output = self.git_ops.diff(cached=True)
        self.assertIn("test_staged.txt", diff_output)

    def test_log_basic(self):
        """Test basic commit log retrieval."""
        for i in range(3):
            with open(f"file{i}.txt", "w") as f:
                f.write(f"content {i}")
            subprocess.run(["git", "add", f"file{i}.txt"], check=True)
            self.git_ops.commit(f"commit {i}")
        
        log_output = self.git_ops.log(max_count=5)
        self.assertIn("commit 0", log_output)
        self.assertIn("commit 2", log_output)

    def test_pull_no_remote(self):
        """Test pull behavior without remote."""
        try:
            self.git_ops.pull(remote="origin", branch="main")
        except RuntimeError as e:
            self.assertIn("fatal", str(e).lower())

if __name__ == "__main__":
    unittest.main()
