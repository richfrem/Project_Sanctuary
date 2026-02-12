#!/usr/bin/env python3
"""
verify_workflow_state.py

A programmatic integrity checker for the Spec Kitty workflow.
Ensures that artifacts (spec, plan, tasks, worktrees) actually exist on disk
before allowing the Agent to proceed.

Usage:
  python3 verify_workflow_state.py --feature <slug> --phase <specify|plan|tasks>
  python3 verify_workflow_state.py --wp <wp-id> --phase <implement|review>
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class VerificationError(Exception):
    """Raised when a workflow integrity check fails."""


def fail(msg: str) -> None:
    """Raise a VerificationError with the given message."""
    raise VerificationError(msg)


def pass_check(msg: str) -> None:
    """Print a success message to stdout."""
    print(f"✅ {msg}")


def find_feature_dir(root: Path, slug: str) -> Path | None:
    """Look in kitty-specs for a directory matching the slug."""
    specs_dir = root / "kitty-specs"
    if not specs_dir.exists():
        fail(f"kitty-specs directory not found at {specs_dir}")

    for p in specs_dir.iterdir():
        if p.is_dir() and (p.name == slug or p.name.endswith(f"-{slug}")):
            return p
    return None


def verify_feature_phase(root: Path, slug: str, phase: str) -> None:
    """Verify that required artifacts exist for a given feature phase."""
    feature_dir = find_feature_dir(root, slug)
    if not feature_dir:
        fail(f"Feature '{slug}' not found in kitty-specs/")

    pass_check(f"Feature directory found: {feature_dir.name}")

    if phase == "specify":
        spec_file = feature_dir / "spec.md"
        if not spec_file.exists():
            fail(f"Missing spec.md in {feature_dir}. Run /spec-kitty.specify first.")
        pass_check("spec.md exists")

    elif phase == "plan":
        plan_file = feature_dir / "plan.md"
        if not plan_file.exists():
            fail(f"Missing plan.md in {feature_dir}. Run /spec-kitty.plan first.")
        pass_check("plan.md exists")

    elif phase == "tasks":
        tasks_main = feature_dir / "tasks.md"
        tasks_dir = feature_dir / "tasks"
        if not tasks_main.exists():
            fail(f"Missing tasks.md in {feature_dir}. Run /spec-kitty.tasks first.")
        if not tasks_dir.exists() or not list(tasks_dir.glob("WP-*.md")):
            fail(f"Missing WP definitions in {tasks_dir}. Run /spec-kitty.tasks first.")
        pass_check("tasks.md and WP files exist")


def _find_worktree(root: Path, wp_id: str) -> Path | None:
    """Find a worktree directory matching the given WP ID."""
    worktrees_dir = root / ".worktrees"
    if not worktrees_dir.exists():
        return None
    for wt in worktrees_dir.iterdir():
        if wt.is_dir() and wt.name.endswith(f"-{wp_id}"):
            return wt
    return None


def _is_worktree_clean(worktree_path: Path) -> bool:
    """Check if a git worktree has no uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == ""
    except (subprocess.SubprocessError, OSError):
        return False


def verify_wp_phase(root: Path, wp_id: str, phase: str) -> None:
    """Verify that required state exists for a given work-package phase."""
    target_wt = _find_worktree(root, wp_id)

    if phase == "implement":
        if not target_wt:
            fail(f"Worktree for {wp_id} not found. Run /spec-kitty.implement {wp_id} first.")
        pass_check(f"Worktree found: {target_wt.name}")

        if not (target_wt / ".git").exists() and not (target_wt / ".git").is_file():
            fail(f"Worktree {target_wt} is not a valid git repository.")
        pass_check("Worktree is a valid git repo")

    elif phase == "review":
        if not target_wt:
            fail(f"Worktree for {wp_id} not found. Cannot review without a worktree.")
        pass_check(f"Worktree found: {target_wt.name}")

        if not _is_worktree_clean(target_wt):
            fail(f"Worktree {target_wt.name} has uncommitted changes. Commit before review.")
        pass_check("Worktree is clean (no uncommitted changes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Spec Kitty Workflow Integrity")
    parser.add_argument("--feature", help="Feature slug (e.g. dual-loop-arch)")
    parser.add_argument("--wp", help="Work Package ID (e.g. WP-001)")
    parser.add_argument("--phase", required=True, choices=["specify", "plan", "tasks", "implement", "review"])

    args = parser.parse_args()
    root = Path(os.getcwd())

    try:
        if args.feature:
            verify_feature_phase(root, args.feature, args.phase)
        elif args.wp:
            verify_wp_phase(root, args.wp, args.phase)
        else:
            fail("Must provide --feature or --wp")
    except VerificationError as e:
        print(f"❌ INTEGRITY FAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()