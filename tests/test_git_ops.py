import unittest
import os
import shutil
import tempfile
import subprocess
from mcp_servers.lib.git.git_ops import GitOperations

class TestGitOperations(unittest.TestCase):
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
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
        
        self.git_ops = GitOperations(self.test_dir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir)

    def test_generate_manifest(self):
        # Create a file
        with open("test.txt", "w") as f:
            f.write("hello world")
            
        # Stage it
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        # Generate manifest
        manifest = self.git_ops.generate_manifest()
        
        self.assertEqual(manifest["author"], "Guardian (Smart Git MCP)")
        self.assertEqual(len(manifest["files"]), 1)
        self.assertEqual(manifest["files"][0]["path"], "test.txt")
        # SHA256 of "hello world"
        expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        self.assertEqual(manifest["files"][0]["sha256"], expected_hash)

    def test_commit_creates_manifest_file(self):
        # Create a file
        with open("test.txt", "w") as f:
            f.write("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        # Commit
        self.git_ops.commit("test commit")
        
        # Check if manifest exists in the commit
        result = subprocess.run(["git", "show", "HEAD:commit_manifest.json"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9", result.stdout)

    def test_status(self):
        # Create a file
        with open("test.txt", "w") as f:
            f.write("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        status = self.git_ops.status()
        self.assertEqual(status["branch"], "main")
        self.assertIn("test.txt", status["staged"])

    def test_branch_operations(self):
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

    def test_push_with_no_verify(self):
        """Test push with no_verify flag."""
        # Create a bare repo to act as remote
        bare_repo = tempfile.mkdtemp()
        subprocess.run(["git", "init", "--bare"], cwd=bare_repo, check=True, capture_output=True)
        
        # Add bare repo as remote
        subprocess.run(["git", "remote", "add", "test-remote", bare_repo], cwd=self.test_dir, check=True)
        
        # Create and commit a file
        with open("test.txt", "w") as f:
            f.write("test content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        self.git_ops.commit("test commit")
        
        # Push with no_verify should succeed
        result = self.git_ops.push("test-remote", "main", no_verify=True)
        self.assertIsNotNone(result)
        
        # Cleanup
        shutil.rmtree(bare_repo)

    def test_push_with_force(self):
        """Test push with force flag."""
        # Create a bare repo to act as remote
        bare_repo = tempfile.mkdtemp()
        subprocess.run(["git", "init", "--bare"], cwd=bare_repo, check=True, capture_output=True)
        
        # Add bare repo as remote
        subprocess.run(["git", "remote", "add", "test-remote", bare_repo], cwd=self.test_dir, check=True)
        
        # Create and commit a file
        with open("test.txt", "w") as f:
            f.write("test content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        self.git_ops.commit("test commit")
        
        # Push normally first
        self.git_ops.push("test-remote", "main", no_verify=True)
        
        # Amend commit to create divergence
        with open("test.txt", "w") as f:
            f.write("modified content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        subprocess.run(["git", "commit", "--amend", "--no-edit"], check=True, capture_output=True)
        
        # Push with force should succeed
        result = self.git_ops.push("test-remote", "main", force=True, no_verify=True)
        self.assertIsNotNone(result)
        
        # Cleanup
        shutil.rmtree(bare_repo)

    def test_push_with_both_flags(self):
        """Test push with both force and no_verify flags."""
        # Create a bare repo to act as remote
        bare_repo = tempfile.mkdtemp()
        subprocess.run(["git", "init", "--bare"], cwd=bare_repo, check=True, capture_output=True)
        
        # Add bare repo as remote
        subprocess.run(["git", "remote", "add", "test-remote", bare_repo], cwd=self.test_dir, check=True)
        
        # Create and commit a file
        with open("test.txt", "w") as f:
            f.write("test content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        self.git_ops.commit("test commit")
        
        # Push with both flags should succeed
        result = self.git_ops.push("test-remote", "main", force=True, no_verify=True)
        self.assertIsNotNone(result)
        
        # Cleanup
        shutil.rmtree(bare_repo)

if __name__ == "__main__":
    unittest.main()
