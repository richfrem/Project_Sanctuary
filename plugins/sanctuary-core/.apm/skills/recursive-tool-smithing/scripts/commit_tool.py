"""
commit_tool.py
==============
Purpose:
    Commits a passing toolbox script to git with a structured commit message.
    Enforces the Graduated Autonomy rules: auto-commit small standalone tools,
    halt for human gate when modifying existing tools or adding dependencies.

Layer: Plugin Utility Script
Plugin: recursive-tool-smithing

Usage:
    python3 commit_tool.py --script scripts/toolbox/pdf_parser.py
                           --gap-slug pdf-parser
                           --rationale "Parse PDFs to structured dict"
                           --metric-before "45s" --metric-after "12s"
    python3 commit_tool.py --help

Output:
    - Prints commit hash on success
    - Prints HUMAN_GATE_REQUIRED if graduated autonomy blocks auto-commit
    - Exits non-zero on git error

Key Functions:
    - check_autonomy_level(script_path, requires_new_dep) -> str
    - build_commit_message(gap_slug, rationale, metric_before, metric_after, goal) -> str
    - commit_script(script_path, message) -> str
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


# Determine whether this commit can proceed autonomously or needs human approval.
def check_autonomy_level(script_path: str, requires_new_dep: bool = False) -> str:
    """Classify commit autonomy level per Graduated Autonomy rules.

    Args:
        script_path: Path to the candidate script.
        requires_new_dep: True if this script adds a new package dependency.

    Returns:
        'auto' if safe to commit without human approval.
        'human_gate' if the change requires explicit user approval.
    """
    path = Path(script_path)
    if not path.exists():
        return "human_gate"

    line_count = len(path.read_text(encoding="utf-8").splitlines())

    if requires_new_dep:
        return "human_gate"
    if line_count > 100:
        return "human_gate"
    return "auto"


# Build a structured commit message following the plugin convention.
def build_commit_message(
    gap_slug: str,
    rationale: str,
    metric_before: str,
    metric_after: str,
    goal: str = "",
) -> str:
    """Construct a conventional commit message for a toolbox addition.

    Args:
        gap_slug: Kebab-case identifier for the gap being resolved.
        rationale: One-line explanation of what the script does.
        metric_before: Baseline metric value (e.g. '45s').
        metric_after: Post-commit metric value (e.g. '12s').
        goal: Strategy goal from program.md (optional).

    Returns:
        Formatted commit message string.
    """
    lines = [
        f"feat(toolbox): add {gap_slug} - {rationale}",
        "",
    ]
    if goal:
        lines.append(f"Strategy: {goal}")
    lines.append(f"Eval: pass ({gap_slug}: {metric_before} -> {metric_after})")
    return "\n".join(lines)


# Stage and commit a toolbox script using git.
def commit_script(script_path: str, message: str) -> str:
    """Stage the script and create a git commit.

    Args:
        script_path: Relative path to the script file.
        message: Commit message string.

    Returns:
        Git commit hash (short) on success.

    Raises:
        RuntimeError: If git add or git commit fails.
    """
    add_result = subprocess.run(
        ["git", "add", script_path],
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        raise RuntimeError(f"git add failed: {add_result.stderr}")

    commit_result = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit_result.stderr}")

    # Extract short hash from output
    lines = commit_result.stdout.strip().splitlines()
    return lines[0] if lines else "unknown"


def main() -> None:
    """CLI entry point for commit_tool."""
    parser = argparse.ArgumentParser(description="Commit a passing toolbox script.")
    parser.add_argument("--script", required=True, help="Path to toolbox script")
    parser.add_argument("--gap-slug", required=True, help="Kebab-case gap identifier")
    parser.add_argument("--rationale", required=True, help="One-line rationale")
    parser.add_argument("--metric-before", default="N/A", help="Baseline metric")
    parser.add_argument("--metric-after", default="N/A", help="Post-commit metric")
    parser.add_argument("--goal", default="", help="Strategy goal from program.md")
    parser.add_argument("--new-dep", action="store_true", help="Set if adding a new dependency")
    parser.add_argument("--dry-run", action="store_true", help="Print commit message without committing")
    args = parser.parse_args()

    autonomy = check_autonomy_level(args.script, requires_new_dep=args.new_dep)
    if autonomy == "human_gate":
        print("HUMAN_GATE_REQUIRED: This change requires explicit approval before committing.")
        print(f"  Script  : {args.script}")
        print(f"  Reason  : large script (>100 lines) or new dependency added")
        return

    message = build_commit_message(
        gap_slug=args.gap_slug,
        rationale=args.rationale,
        metric_before=args.metric_before,
        metric_after=args.metric_after,
        goal=args.goal,
    )

    if args.dry_run:
        print("--- DRY RUN: Commit message ---")
        print(message)
        return

    try:
        commit_hash = commit_script(args.script, message)
        print(f"Committed: {commit_hash}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)
    # Self-test (no git side effects)
    msg = build_commit_message(
        gap_slug="pdf-parser",
        rationale="Parse PDFs to structured dict",
        metric_before="45s",
        metric_after="12s",
        goal="Minimize query latency",
    )
    assert "feat(toolbox): add pdf-parser" in msg
    assert "45s -> 12s" in msg
    assert "Minimize query latency" in msg

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1\n" * 50)  # 50 lines -> auto
        tmp_50 = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1\n" * 150)  # 150 lines -> human_gate
        tmp_150 = f.name

    assert check_autonomy_level(tmp_50) == "auto"
    assert check_autonomy_level(tmp_150) == "human_gate"
    assert check_autonomy_level(tmp_50, requires_new_dep=True) == "human_gate"

    os.unlink(tmp_50)
    os.unlink(tmp_150)
    print("commit_tool: all self-tests passed.")
