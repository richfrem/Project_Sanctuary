import unittest
import os
import shutil
import tempfile
import subprocess
from mcp_servers.lib.git.git_ops import GitOperations

class TestGitOperations(unittest.TestCase):
    """
    Test suite for GitOperations class (Protocol 101 v3.0 compliant).
    
    Note: Manifest generation tests have been removed as Protocol 101 v3.0
    uses Functional Coherence (test suite execution) instead of manifests.
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
        """Test repository status retrieval."""
        # Create a file
        with open("test.txt", "w") as f:
            f.write("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        status = self.git_ops.status()
        self.assertEqual(status["branch"], "main")
        self.assertIn("test.txt", status["staged"])

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

if __name__ == "__main__":
    unittest.main()
