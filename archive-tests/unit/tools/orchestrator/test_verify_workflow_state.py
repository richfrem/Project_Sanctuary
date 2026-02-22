"""
test_verify_workflow_state.py

Unit tests for tools/orchestrator/verify_workflow_state.py.
Uses tempfile to create isolated directory structures for each test.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.orchestrator.verify_workflow_state import (
    VerificationError,
    _find_worktree,
    _is_worktree_clean,
    find_feature_dir,
    verify_feature_phase,
    verify_wp_phase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature(root: Path, slug: str, files: list[str] | None = None) -> Path:
    """Create a feature directory under kitty-specs/ with optional files."""
    feature_dir = root / "kitty-specs" / slug
    feature_dir.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        path = feature_dir / f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {f}")
    return feature_dir


def _make_worktree(root: Path, name: str, *, git: bool = True) -> Path:
    """Create a fake worktree directory. If git=True, create a .git marker."""
    wt = root / ".worktrees" / name
    wt.mkdir(parents=True, exist_ok=True)
    if git:
        (wt / ".git").write_text("gitdir: /fake/path")
    return wt


# ---------------------------------------------------------------------------
# find_feature_dir
# ---------------------------------------------------------------------------

class TestFindFeatureDir:
    def test_missing_kitty_specs_dir(self, tmp_path: Path):
        with pytest.raises(VerificationError, match="kitty-specs directory not found"):
            find_feature_dir(tmp_path, "anything")

    def test_exact_match(self, tmp_path: Path):
        _make_feature(tmp_path, "my-feature")
        result = find_feature_dir(tmp_path, "my-feature")
        assert result is not None
        assert result.name == "my-feature"

    def test_suffix_match(self, tmp_path: Path):
        _make_feature(tmp_path, "004-verify-workflow-test")
        result = find_feature_dir(tmp_path, "verify-workflow-test")
        assert result is not None
        assert result.name == "004-verify-workflow-test"

    def test_no_match(self, tmp_path: Path):
        (tmp_path / "kitty-specs").mkdir()
        result = find_feature_dir(tmp_path, "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# verify_feature_phase — specify
# ---------------------------------------------------------------------------

class TestVerifySpecifyPhase:
    def test_missing_feature_dir(self, tmp_path: Path):
        (tmp_path / "kitty-specs").mkdir()
        with pytest.raises(VerificationError, match="not found in kitty-specs"):
            verify_feature_phase(tmp_path, "ghost", "specify")

    def test_missing_spec_md(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-a")
        with pytest.raises(VerificationError, match="Missing spec.md"):
            verify_feature_phase(tmp_path, "feat-a", "specify")

    def test_success(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-a", ["spec.md"])
        verify_feature_phase(tmp_path, "feat-a", "specify")


# ---------------------------------------------------------------------------
# verify_feature_phase — plan
# ---------------------------------------------------------------------------

class TestVerifyPlanPhase:
    def test_missing_plan_md(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-b", ["spec.md"])
        with pytest.raises(VerificationError, match="Missing plan.md"):
            verify_feature_phase(tmp_path, "feat-b", "plan")

    def test_success(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-b", ["spec.md", "plan.md"])
        verify_feature_phase(tmp_path, "feat-b", "plan")


# ---------------------------------------------------------------------------
# verify_feature_phase — tasks
# ---------------------------------------------------------------------------

class TestVerifyTasksPhase:
    def test_missing_tasks_md(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-c", ["spec.md", "plan.md"])
        with pytest.raises(VerificationError, match="Missing tasks.md"):
            verify_feature_phase(tmp_path, "feat-c", "tasks")

    def test_missing_wp_files(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-c", ["spec.md", "plan.md", "tasks.md"])
        with pytest.raises(VerificationError, match="Missing WP definitions"):
            verify_feature_phase(tmp_path, "feat-c", "tasks")

    def test_success(self, tmp_path: Path):
        _make_feature(tmp_path, "feat-c", ["spec.md", "plan.md", "tasks.md", "tasks/WP-001.md"])
        verify_feature_phase(tmp_path, "feat-c", "tasks")


# ---------------------------------------------------------------------------
# _find_worktree
# ---------------------------------------------------------------------------

class TestFindWorktree:
    def test_no_worktrees_dir(self, tmp_path: Path):
        assert _find_worktree(tmp_path, "WP-001") is None

    def test_no_match(self, tmp_path: Path):
        (tmp_path / ".worktrees" / "other-WP-999").mkdir(parents=True)
        assert _find_worktree(tmp_path, "WP-001") is None

    def test_match(self, tmp_path: Path):
        _make_worktree(tmp_path, "feat-x-WP-001")
        result = _find_worktree(tmp_path, "WP-001")
        assert result is not None
        assert result.name == "feat-x-WP-001"


# ---------------------------------------------------------------------------
# verify_wp_phase — implement
# ---------------------------------------------------------------------------

class TestVerifyImplementPhase:
    def test_missing_worktree(self, tmp_path: Path):
        with pytest.raises(VerificationError, match="Worktree for WP-001 not found"):
            verify_wp_phase(tmp_path, "WP-001", "implement")

    def test_missing_git(self, tmp_path: Path):
        _make_worktree(tmp_path, "feat-WP-001", git=False)
        with pytest.raises(VerificationError, match="not a valid git repository"):
            verify_wp_phase(tmp_path, "WP-001", "implement")

    def test_success(self, tmp_path: Path):
        _make_worktree(tmp_path, "feat-WP-001", git=True)
        verify_wp_phase(tmp_path, "WP-001", "implement")


# ---------------------------------------------------------------------------
# verify_wp_phase — review
# ---------------------------------------------------------------------------

class TestVerifyReviewPhase:
    def test_missing_worktree(self, tmp_path: Path):
        with pytest.raises(VerificationError, match="Cannot review without a worktree"):
            verify_wp_phase(tmp_path, "WP-001", "review")

    def test_dirty_worktree(self, tmp_path: Path):
        _make_worktree(tmp_path, "feat-WP-001")
        with patch("tools.orchestrator.verify_workflow_state._is_worktree_clean", return_value=False):
            with pytest.raises(VerificationError, match="uncommitted changes"):
                verify_wp_phase(tmp_path, "WP-001", "review")

    def test_clean_worktree(self, tmp_path: Path):
        _make_worktree(tmp_path, "feat-WP-001")
        with patch("tools.orchestrator.verify_workflow_state._is_worktree_clean", return_value=True):
            verify_wp_phase(tmp_path, "WP-001", "review")


# ---------------------------------------------------------------------------
# _is_worktree_clean
# ---------------------------------------------------------------------------

class TestIsWorktreeClean:
    def test_clean(self, tmp_path: Path):
        with patch("tools.orchestrator.verify_workflow_state.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            assert _is_worktree_clean(tmp_path) is True

    def test_dirty(self, tmp_path: Path):
        with patch("tools.orchestrator.verify_workflow_state.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=" M file.txt\n", stderr=""
            )
            assert _is_worktree_clean(tmp_path) is False

    def test_git_error(self, tmp_path: Path):
        with patch("tools.orchestrator.verify_workflow_state.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=128, stdout="", stderr="fatal: not a git repo"
            )
            assert _is_worktree_clean(tmp_path) is False

    def test_subprocess_exception(self, tmp_path: Path):
        with patch("tools.orchestrator.verify_workflow_state.subprocess.run", side_effect=OSError("no git")):
            assert _is_worktree_clean(tmp_path) is False
