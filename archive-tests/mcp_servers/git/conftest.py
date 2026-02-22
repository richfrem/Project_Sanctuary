import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def git_roots(tmp_path):
    """Create a remote (bare) and local repository pair."""
    remote_path = tmp_path / "remote.git"
    local_path = tmp_path / "local"
    
    # Initialize bare remote
    import subprocess
    subprocess.run(["git", "init", "--bare", str(remote_path)], check=True, capture_output=True)
    
    # Clone local
    subprocess.run(["git", "clone", str(remote_path), str(local_path)], check=True, capture_output=True)
    
    # Configure local
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(local_path), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(local_path), check=True)
    
    # Initial commit (needs to be pushed to remote so 'main' exists there)
    (local_path / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "README.md"], cwd=str(local_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(local_path), check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(local_path), check=True)
    
    return {"remote": remote_path, "local": local_path}

@pytest.fixture
def git_root(tmp_path):
    """Create a temporary directory for Git tests (legacy single-repo support)."""
    root = tmp_path / "git_test_root"
    root.mkdir()
    return root

@pytest.fixture
def git_ops_mock(git_root):
    """Create GitOperations instance with local root (legacy)."""
    from mcp_servers.git.operations import GitOperations
    ops = GitOperations(git_root)
    return ops

@pytest.fixture
def git_ops_with_remote(git_roots):
    """Create GitOperations instance connected to a local 'remote'."""
    from mcp_servers.git.operations import GitOperations
    ops = GitOperations(git_roots["local"])
    return ops
