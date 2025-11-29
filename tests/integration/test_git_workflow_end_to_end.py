import unittest
import os
import shutil
import tempfile
import subprocess
from mcp_servers.lib.git.git_ops import GitOperations


class TestGitWorkflowEndToEnd(unittest.TestCase):
    """
    Integration test for complete Git workflow.
    Tests the full cycle: branch → commit → push → cleanup
    """

    def setUp(self):
        """Set up test environment with local and bare repos."""
        # Create temporary directory for test repo
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
        
        # Create bare repo to act as remote
        self.bare_repo = tempfile.mkdtemp()
        subprocess.run(["git", "init", "--bare"], cwd=self.bare_repo, check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", self.bare_repo], cwd=self.test_dir, check=True)
        
        # Push main to remote
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True, capture_output=True)
        
        self.git_ops = GitOperations(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.bare_repo)

    def test_complete_feature_workflow(self):
        """
        Test complete feature workflow:
        1. Create feature branch
        2. Make changes
        3. Stage files
        4. Commit with manifest
        5. Push with no_verify=True
        6. Verify commit exists
        7. Cleanup
        """
        # 1. Create feature branch
        feature_branch = "feature/test-workflow"
        self.git_ops.create_branch(feature_branch)
        self.git_ops.checkout(feature_branch)
        self.assertEqual(self.git_ops.get_current_branch(), feature_branch)
        
        # 2. Make changes
        with open("feature.txt", "w") as f:
            f.write("Feature implementation")
        
        # 3. Stage files
        self.git_ops.add(["feature.txt"])
        
        # 4. Commit with manifest (Protocol 101)
        commit_hash = self.git_ops.commit("feat: add feature implementation")
        self.assertIsNotNone(commit_hash)
        self.assertEqual(len(commit_hash), 40)  # SHA-1 hash length
        
        # Verify manifest was created
        result = subprocess.run(
            ["git", "show", "HEAD:commit_manifest.json"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("feature.txt", result.stdout)
        
        # 5. Push with no_verify=True (bypasses git-lfs hook)
        push_result = self.git_ops.push("origin", feature_branch, no_verify=True)
        self.assertIsNotNone(push_result)
        
        # 6. Verify commit exists on remote
        # Clone the bare repo to verify
        verify_dir = tempfile.mkdtemp()
        subprocess.run(
            ["git", "clone", self.bare_repo, verify_dir],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "checkout", feature_branch],
            cwd=verify_dir,
            check=True,
            capture_output=True
        )
        
        # Check that feature.txt exists in the cloned repo
        feature_file = os.path.join(verify_dir, "feature.txt")
        self.assertTrue(os.path.exists(feature_file))
        
        with open(feature_file, "r") as f:
            content = f.read()
        self.assertEqual(content, "Feature implementation")
        
        # Cleanup verify dir
        shutil.rmtree(verify_dir)
        
        # 7. Cleanup - switch back to main and delete feature branch
        self.git_ops.checkout("main")
        self.git_ops.delete_branch(feature_branch, force=True)  # Force delete since not merged

    def test_workflow_with_force_push(self):
        """
        Test workflow with force push after amending commit.
        """
        # Create feature branch
        feature_branch = "feature/test-force"
        self.git_ops.create_branch(feature_branch)
        self.git_ops.checkout(feature_branch)
        
        # Create and commit file
        with open("test.txt", "w") as f:
            f.write("original content")
        self.git_ops.add(["test.txt"])
        self.git_ops.commit("feat: add test file")
        
        # Push to remote
        self.git_ops.push("origin", feature_branch, no_verify=True)
        
        # Amend commit (creates divergence)
        with open("test.txt", "w") as f:
            f.write("modified content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        subprocess.run(["git", "commit", "--amend", "--no-edit"], check=True, capture_output=True)
        
        # Force push should succeed
        result = self.git_ops.push("origin", feature_branch, force=True, no_verify=True)
        self.assertIsNotNone(result)
        
        # Cleanup
        self.git_ops.checkout("main")
        self.git_ops.delete_branch(feature_branch, force=True)  # Force delete since not merged


if __name__ == "__main__":
    unittest.main()
