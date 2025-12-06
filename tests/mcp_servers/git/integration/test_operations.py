import pytest
import subprocess
import os
from mcp_servers.git.git_ops import GitOperations

class TestGitOperationsIntegration:
    """
    Test suite for GitOperations class (Protocol 101 v3.0 compliant).
    Converted from unittest to pytest.
    """

    @pytest.fixture(autouse=True)
    def setup_repo(self, git_root, monkeypatch):
        """Initialize a git repo for each test."""
        # Using monkeypatch to change cwd safely during test
        monkeypatch.chdir(git_root)
        
        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        
        # Create initial commit
        (git_root / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "README.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit", "--no-verify"], check=True)

    def test_commit_basic(self, git_ops_mock, git_root):
        """Test basic commit functionality."""
        (git_root / "test.txt").write_text("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        commit_hash = git_ops_mock.commit("test commit")
        
        assert commit_hash is not None
        assert len(commit_hash) == 40

    def test_add_files(self, git_ops_mock, git_root):
        """Test adding specific files."""
        (git_root / "file1.txt").write_text("content1")
        (git_root / "file2.txt").write_text("content2")
        
        # Verify initially untracked
        status = git_ops_mock.status()
        assert "file1.txt" in status["untracked"]
        
        # Add specific file
        git_ops_mock.add(["file1.txt"])
        
        status = git_ops_mock.status()
        assert "file1.txt" in status["staged"]
        assert "file2.txt" in status["untracked"]

    def test_add_all(self, git_ops_mock, git_root):
        """Test adding all files (git add -A)."""
        (git_root / "file1.txt").write_text("content1")
        (git_root / "file2.txt").write_text("content2")
        
        git_ops_mock.add()  # None implies -A
        
        status = git_ops_mock.status()
        assert "file1.txt" in status["staged"]
        assert "file2.txt" in status["staged"]

    def test_status(self, git_ops_mock, git_root):
        """Test repository status retrieval."""
        (git_root / "test.txt").write_text("hello world")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        status = git_ops_mock.status()
        
        assert status["branch"] == "main"
        assert "test.txt" in status["staged"]
        assert "local_branches" in status
        assert "is_clean" in status
        assert status["is_clean"] is False

    def test_branch_operations(self, git_ops_mock):
        """Test branch creation, checkout, and deletion."""
        git_ops_mock.create_branch("feature/test")
        
        git_ops_mock.checkout("feature/test")
        assert git_ops_mock.get_current_branch() == "feature/test"
        
        git_ops_mock.checkout("main")
        assert git_ops_mock.get_current_branch() == "main"
        
        git_ops_mock.delete_branch("feature/test")
        
        with pytest.raises(RuntimeError):
            git_ops_mock.checkout("feature/test")

    def test_get_staged_files(self, git_ops_mock, git_root):
        """Test retrieval of staged files."""
        (git_root / "test.txt").write_text("content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        
        staged = git_ops_mock.get_staged_files()
        assert "test.txt" in staged

    def test_push_with_no_verify(self, git_ops_mock, git_root):
        """Test push with no_verify parameter."""
        (git_root / "test.txt").write_text("test content")
        subprocess.run(["git", "add", "test.txt"], check=True)
        git_ops_mock.commit("test commit for push")
        
        try:
            git_ops_mock.push(remote="origin", no_verify=True)
        except RuntimeError as e:
            assert "fatal" in str(e).lower()

    def test_diff_unstaged(self, git_ops_mock, git_root):
        """Test diff for unstaged changes."""
        test_file = git_root / "test_diff.txt"
        test_file.write_text("original content")
        subprocess.run(["git", "add", "test_diff.txt"], check=True)
        git_ops_mock.commit("add test_diff.txt")
        
        test_file.write_text("modified content")
        
        diff_output = git_ops_mock.diff(cached=False)
        assert "test_diff.txt" in diff_output

    def test_diff_staged(self, git_ops_mock, git_root):
        """Test diff for staged changes."""
        test_file = git_root / "test_staged.txt"
        test_file.write_text("staged content")
        subprocess.run(["git", "add", "test_staged.txt"], check=True)
        
        diff_output = git_ops_mock.diff(cached=True)
        assert "test_staged.txt" in diff_output

    def test_log_basic(self, git_ops_mock, git_root):
        """Test basic commit log retrieval."""
        for i in range(3):
            file_name = f"file{i}.txt"
            (git_root / file_name).write_text(f"content {i}")
            subprocess.run(["git", "add", file_name], check=True)
            git_ops_mock.commit(f"commit {i}")
        
        log_output = git_ops_mock.log(max_count=5)
        assert "commit 0" in log_output
        assert "commit 2" in log_output

    def test_pull_no_remote(self, git_ops_mock):
        """Test pull behavior without remote."""
        try:
            git_ops_mock.pull(remote="origin", branch="main")
        except RuntimeError as e:
            assert "fatal" in str(e).lower()

    def test_finish_feature_success(self, git_ops_with_remote, git_roots):
        """test_finish_feature_success: Verify successful cleanup of merged branch."""
        ops = git_ops_with_remote
        feature_branch = "feature/test-task"
        
        # 1. Create and checkout feature
        ops.create_branch(feature_branch)
        ops.checkout(feature_branch)
        
        # 2. Make change and commit
        (git_roots["local"] / "feature.txt").write_text("feature content")
        ops.add(["feature.txt"])
        ops.commit("feature commit")
        
        # 3. Push to remote
        ops.push("origin", feature_branch)
        
        # 4. Simulate Merge on Remote (by checking out main on local proxy, merging, and pushing)
        # We simulate what GitHub PR merge does
        ops.checkout("main")
        subprocess.run(["git", "merge", feature_branch], cwd=str(git_roots["local"]), check=True)
        ops.push("origin", "main")
        
        # 5. Switch back to feature branch (typical user state before fishing)
        ops.checkout(feature_branch)
        
        # 6. Run finish_feature
        result = ops.finish_feature(feature_branch)
        
        assert "Finished feature" in result
        assert ops.get_current_branch() == "main"
        
        # Verify deletion
        with pytest.raises(RuntimeError):
            ops.checkout(feature_branch)

    def test_finish_feature_unmerged_failure(self, git_ops_with_remote, git_roots):
        """test_finish_feature_unmerged_failure: Ensure unmerged branch triggers error."""
        ops = git_ops_with_remote
        feature_branch = "feature/unmerged"
        
        ops.create_branch(feature_branch)
        ops.checkout(feature_branch)
        
        (git_roots["local"] / "wip.txt").write_text("wip")
        ops.add(["wip.txt"])
        ops.commit("wip")
        
        # Attempt finish without merge
        with pytest.raises(RuntimeError) as excinfo:
            ops.finish_feature(feature_branch)
        
        assert "NOT merged into main" in str(excinfo.value)

    def test_finish_feature_squash_detection(self, git_ops_with_remote, git_roots):
        """test_finish_feature_squash_detection: Verify squash merge (diff check) logic."""
        ops = git_ops_with_remote
        feature_branch = "feature/squash-test"
        
        ops.create_branch(feature_branch)
        ops.checkout(feature_branch)
        
        (git_roots["local"] / "squash.txt").write_text("squashed")
        ops.add(["squash.txt"])
        ops.commit("squash commit")
        ops.push("origin", feature_branch)
        
        # Simulate Squash Merge on Remote:
        # Switch main and create IDENTICAL content manually (squash effect)
        ops.checkout("main")
        (git_roots["local"] / "squash.txt").write_text("squashed")
        ops.add(["squash.txt"])
        ops.commit("Squash merge commit manually")
        ops.push("origin", "main")
        
        # Switch back
        ops.checkout(feature_branch)
        
        # Run finish - should succeed due to diff check
        result = ops.finish_feature(feature_branch)
        assert "Finished feature" in result

    def test_full_feature_lifecycle(self, git_ops_with_remote, git_roots):
        """
        Ordered Integration Test: Verify the exact user workflow sequence.
        Lifecycle: Start -> Edit -> Diff -> Add -> Status -> Commit -> Log -> Push -> Finish
        """
        ops = git_ops_with_remote
        feature_branch = "feature/lifecycle-test"
        
        # 1. Start Feature (git_start_feature)
        ops.create_branch(feature_branch)
        ops.checkout(feature_branch)
        assert ops.get_current_branch() == feature_branch
        
        # 2. Edit (Modify existing file so diff picks it up)
        (git_roots["local"] / "README.md").write_text("# Test Repo\n\nModified content")
        
        # 3. Diff Unstaged (git_diff)
        diff = ops.diff(cached=False)
        assert "README.md" in diff
        assert "+Modified content" in diff
        
        # 4. Add (git_add)
        ops.add(["README.md"])
        
        # 5. Verify Staged (Status/Diff)
        diff_cached = ops.diff(cached=True)
        assert "README.md" in diff_cached
        status = ops.status()
        assert "README.md" in status["staged"]
        
        # 6. Commit (git_smart_commit)
        ops.commit("feat: lifecycle test")
        
        # 7. Log (git_log)
        log = ops.log(max_count=1)
        assert "feat: lifecycle test" in log
        
        # 8. Push Feature (git_push_feature)
        ops.push("origin", feature_branch)
        
        # 9. Finish Feature (git_finish_feature)
        # Prerequisite: Simulate GitHub PR Merge on Remote
        ops.checkout("main")
        subprocess.run(["git", "merge", feature_branch], cwd=str(git_roots["local"]), check=True)
        ops.push("origin", "main")
        
        # Return to feature branch to trigger finish
        ops.checkout(feature_branch)
        result = ops.finish_feature(feature_branch)
        
        assert "Finished feature" in result
        assert ops.get_current_branch() == "main"
        
        # Verify complete cleanup
        with pytest.raises(RuntimeError):
            ops.checkout(feature_branch)

