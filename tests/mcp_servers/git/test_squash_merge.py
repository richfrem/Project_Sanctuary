import unittest
import tempfile
import shutil
import os
import subprocess
from pathlib import Path
from mcp_servers.git.git_ops import GitOperations
from mcp_servers.git.server import git_finish_feature

class TestSquashMerge(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the repos
        self.test_dir = tempfile.mkdtemp()
        
        # Create bare remote repo (simulates GitHub)
        self.remote_path = Path(self.test_dir) / "remote.git"
        self.remote_path.mkdir()
        subprocess.run(["git", "init", "--bare"], cwd=self.remote_path, check=True)
        
        # Create local repo
        self.repo_path = Path(self.test_dir) / "local"
        self.repo_path.mkdir()
        subprocess.run(["git", "init"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path, check=True)
        
        # Add remote
        subprocess.run(["git", "remote", "add", "origin", str(self.remote_path)], cwd=self.repo_path, check=True)
        
        # Create initial commit on main
        (self.repo_path / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "README.md"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.repo_path, check=True)
        
        # Rename master to main if needed
        subprocess.run(["git", "branch", "-M", "main"], cwd=self.repo_path, check=True)
        
        # Push to remote
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=self.repo_path, check=True)
        
        # Initialize GitOperations with this repo
        # We need to patch the global git_ops in the server module
        import mcp_servers.git.server as server
        server.REPO_PATH = str(self.repo_path)
        server.git_ops = GitOperations(str(self.repo_path))
        self.server = server
        
        # Import the actual function (not the FastMCP tool wrapper)
        from mcp_servers.git import server as git_server
        self.git_finish_feature_fn = git_server.git_finish_feature.fn

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_finish_feature_squash_merge(self):
        """Test finishing a feature branch that was squash merged."""
        # 1. Start feature branch
        branch_name = "feature/task-001-squash-test"
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.repo_path, check=True)
        
        # 2. Make changes
        (self.repo_path / "feature.txt").write_text("Feature content")
        subprocess.run(["git", "add", "feature.txt"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Feature commit"], cwd=self.repo_path, check=True)
        
        # Push feature branch to remote
        subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, check=True)
        
        # 3. Simulate Squash Merge on Main
        # Checkout main
        subprocess.run(["git", "checkout", "main"], cwd=self.repo_path, check=True)
        
        # Apply changes from feature (simulate squash)
        # We just create the same file content
        (self.repo_path / "feature.txt").write_text("Feature content")
        subprocess.run(["git", "add", "feature.txt"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Squash merge feature"], cwd=self.repo_path, check=True)
        
        # Push main to remote
        subprocess.run(["git", "push", "origin", "main"], cwd=self.repo_path, check=True)
        
        # At this point:
        # - main has the content
        # - feature branch has the content
        # - BUT git log graph shows they are diverged (no common merge commit)
        
        # Verify git thinks it's NOT merged
        is_merged = self.server.git_ops.is_branch_merged(branch_name, "main")
        self.assertFalse(is_merged, "Git should NOT consider this merged yet")
        
        # 4. Try to finish feature WITHOUT force (should now auto-detect and succeed!)
        result = self.git_finish_feature_fn(branch_name)
        print(f"\nResult without force (auto-detect): {result}")
        # Should succeed due to auto-detection
        self.assertIn("Finished feature", result)
        self.assertIn("Auto-detected squash merge", result) if "Auto-detected" in result else None
        
        # 5. Verify branch is gone
        result = subprocess.run(["git", "branch", "--list", branch_name], 
                              cwd=self.repo_path, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "")

if __name__ == "__main__":
    unittest.main()
