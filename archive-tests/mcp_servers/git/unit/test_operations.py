"""
Unit tests for Git Operations (Business Logic).
Decoupled from Pydantic Models. Mocks git command execution.
"""
import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
from mcp_servers.git.operations import GitOperations
from mcp_servers.git.models import GitStatus

class TestGitOperations:
    @pytest.fixture
    def setup_ops(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = GitOperations(repo_path=self.test_dir)
        yield
        shutil.rmtree(self.test_dir)

    @patch("subprocess.run")
    def test_status_clean(self, mock_subprocess, setup_ops):
        """Test parsing robust status output."""
        # Mock status --porcelain
        # Mock branch -vv
        # Mock rev-parse (upstream)
        
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            mock_res = MagicMock()
            mock_res.returncode = 0
            
            if "status --porcelain" in cmd_str:
                mock_res.stdout = "" # Clean
            elif "branch -vv" in cmd_str:
                mock_res.stdout = "* main 123456 [origin/main] Commit msg"
            elif "rev-parse --abbrev-ref HEAD" in cmd_str:
                mock_res.stdout = "main"
            elif "rev-parse --abbrev-ref main@{upstream}" in cmd_str:
                mock_res.stdout = "origin/main"
            elif "rev-list" in cmd_str:
                mock_res.stdout = "0 0" # Ahead Behind
            else:
                mock_res.stdout = ""
                
            return mock_res

        mock_subprocess.side_effect = side_effect
        
        status = self.ops.status()
        
        assert status.branch == "main"
        assert status.is_clean
        assert len(status.feature_branches) == 0

    @patch("subprocess.run")
    def test_start_feature(self, mock_subprocess, setup_ops):
        """Test feature branch creation."""
        # Mock status as clean on main
        
        def side_effect(cmd, **kwargs):
            mock_res = MagicMock()
            mock_res.returncode = 0
            cmd_str = " ".join(cmd)
            
            if "status --porcelain" in cmd_str:
                mock_res.stdout = ""
            elif "branch -vv" in cmd_str:
                mock_res.stdout = "* main 123456 [origin/main]"
            elif "rev-parse --abbrev-ref HEAD" in cmd_str:
                mock_res.stdout = "main"
            elif "rev-list" in cmd_str:
                mock_res.stdout = "0 0"
            elif "branch feature/" in cmd_str:
                mock_res.stdout = "" # Create branch
            elif "checkout" in cmd_str:
                mock_res.stdout = "" 
            else:
                mock_res.stdout = ""
            return mock_res

        mock_subprocess.side_effect = side_effect
        
        msg = self.ops.start_feature("123", "Unit Test")
        
        assert "Created and switched" in msg
        assert "feature/task-123-unit-test" in msg
        
        # Verify branch creation call
        calls = [c[0][0] for c in mock_subprocess.call_args_list]
        branch_cmd = [c for c in calls if "branch" in c and "feature/task-123-unit-test" in c]
        assert branch_cmd

    @patch("subprocess.run")
    def test_commit_enforcement(self, mock_subprocess, setup_ops):
        """Test commit calls validator and git commit."""
        def side_effect(cmd, **kwargs):
            mock_res = MagicMock()
            mock_res.returncode = 0
            cmd_str = " ".join(cmd)
            
            if "rev-parse --abbrev-ref HEAD" in cmd_str:
                mock_res.stdout = "feature/task-123-foo" # Valid feature branch
            if "diff --name-only --cached" in cmd_str:
                mock_res.stdout = "file.txt"
            return mock_res
            
        mock_subprocess.side_effect = side_effect
        
        self.ops.commit("Test commit")
        
        # Verify call to commit
        calls = [c[0][0] for c in mock_subprocess.call_args_list]
        commit_cmd = [c for c in calls if "commit" in c and "-m" in c]
        assert commit_cmd

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
