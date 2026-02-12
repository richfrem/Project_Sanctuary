#!/usr/bin/env python3
"""
Dual-Loop Wrapper CLI (Protocol 133 Orchestrator)
==================================================

Orchestrates the Dual-Loop Workflow:
1. Locate feature artifacts (tasks.md, spec.md, plan.md)
2. Generate Strategy Packet
3. Launch Inner Loop (Claude) in worktree isolation
4. (Optional) Run Verification

Usage:
  python3 run_workflow.py <TASK_ID> --feature <SLUG>
  python3 run_workflow.py <TASK_ID> --tasks-file <PATH>

Related:
  - Skill: .agent/skills/dual-loop-supervisor/SKILL.md
  - Protocol 133: Dual-Loop Agent Architecture
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KITTY_SPECS = PROJECT_ROOT / "kitty-specs"
HANDOFFS_DIR = PROJECT_ROOT / ".agent" / "handoffs"


def find_feature_dir(slug: str) -> Path | None:
    """Find a kitty-specs feature directory by slug."""
    if not KITTY_SPECS.exists():
        return None
    for p in KITTY_SPECS.iterdir():
        if p.is_dir() and (p.name == slug or p.name.endswith(f"-{slug}")):
            return p
    return None


def detect_feature_from_branch() -> str | None:
    """Try to extract feature slug from current git branch name.

    Branch format: feat/NNN-slug or NNN-slug.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        branch = result.stdout.strip()
        # Strip common prefixes
        for prefix in ("feat/", "feature/", "fix/", "chore/"):
            if branch.startswith(prefix):
                branch = branch[len(prefix):]
                break
        return branch if branch else None
    except (subprocess.SubprocessError, OSError):
        return None


def resolve_tasks_file(args) -> tuple[Path, Path | None, Path | None]:
    """Resolve tasks.md, spec.md, and plan.md paths from arguments.

    Returns (tasks_path, spec_path, plan_path).
    """
    if args.tasks_file:
        tasks_path = Path(args.tasks_file)
        if not tasks_path.is_absolute():
            tasks_path = PROJECT_ROOT / tasks_path
        feature_dir = tasks_path.parent
        spec_path = feature_dir / "spec.md"
        plan_path = feature_dir / "plan.md"
        return (
            tasks_path,
            spec_path if spec_path.exists() else None,
            plan_path if plan_path.exists() else None,
        )

    # Resolve from --feature or auto-detect from branch
    slug = args.feature
    if not slug:
        slug = detect_feature_from_branch()
        if slug:
            print(f"[INFO] Auto-detected feature from branch: {slug}")

    if not slug:
        print(
            "[ERROR] Cannot determine feature. Use --feature <SLUG> or --tasks-file <PATH>.",
            file=sys.stderr,
        )
        sys.exit(1)

    feature_dir = find_feature_dir(slug)
    if not feature_dir:
        print(f"[ERROR] Feature '{slug}' not found in kitty-specs/.", file=sys.stderr)
        sys.exit(1)

    tasks_path = feature_dir / "tasks.md"
    if not tasks_path.exists():
        print(f"[ERROR] tasks.md not found at {tasks_path}.", file=sys.stderr)
        sys.exit(1)

    spec_path = feature_dir / "spec.md"
    plan_path = feature_dir / "plan.md"
    return (
        tasks_path,
        spec_path if spec_path.exists() else None,
        plan_path if plan_path.exists() else None,
    )


def find_worktree(task_id: str) -> Path | None:
    """Find a worktree directory matching the given task ID."""
    worktrees_dir = PROJECT_ROOT / ".worktrees"
    if not worktrees_dir.exists():
        return None
    for wt in worktrees_dir.iterdir():
        if wt.is_dir() and task_id.lower() in wt.name.lower():
            return wt
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate a Dual-Loop Workflow (Protocol 133).",
        epilog=(
            "Examples:\n"
            "  %(prog)s WP-001 --feature 005-my-feature\n"
            "  %(prog)s WP-001 --tasks-file kitty-specs/005-my-feature/tasks.md\n"
            "  %(prog)s WP-001  # auto-detects feature from git branch"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task_id", help="Task ID (e.g., 'WP-001', '3')")
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Feature slug (e.g., '005-my-feature'). Auto-detected from branch if omitted.",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help="Explicit path to tasks.md (overrides --feature).",
    )
    parser.add_argument(
        "--skip-worktree",
        action="store_true",
        help="Skip worktree creation, generate packet only (Branch-Direct Mode).",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Generate packet but do not launch Claude. Useful for manual hand-off.",
    )
    args = parser.parse_args()

    tasks_path, spec_path, plan_path = resolve_tasks_file(args)
    print(f"[Outer Loop] Dual-Loop Sequence for Task {args.task_id}")
    print(f"[INFO] Tasks: {tasks_path}")
    if spec_path:
        print(f"[INFO] Spec:  {spec_path}")
    if plan_path:
        print(f"[INFO] Plan:  {plan_path}")

    # 1. Worktree setup (optional)
    worktree_dir = None
    if not args.skip_worktree:
        worktree_dir = find_worktree(args.task_id)
        if worktree_dir:
            print(f"[OK] Existing worktree found: {worktree_dir}")
        else:
            print(
                f"[WARN] No worktree found for {args.task_id}. "
                "Run /spec-kitty.implement first, or use --skip-worktree for Branch-Direct Mode."
            )

    # 2. Generate Strategy Packet
    print("[Dual Loop] Generating Strategy Packet...")
    generator_tool = PROJECT_ROOT / "tools/orchestrator/dual_loop/generate_strategy_packet.py"

    cmd_generate = [
        sys.executable,
        str(generator_tool),
        "--tasks-file",
        str(tasks_path),
        "--task-id",
        args.task_id,
    ]
    if spec_path:
        cmd_generate.extend(["--spec", str(spec_path)])
    if plan_path:
        cmd_generate.extend(["--plan", str(plan_path)])

    gen_result = subprocess.run(cmd_generate, cwd=PROJECT_ROOT)
    if gen_result.returncode != 0:
        print("[ERROR] Failed to generate Strategy Packet.", file=sys.stderr)
        sys.exit(1)

    # Find the generated packet
    HANDOFFS_DIR.mkdir(parents=True, exist_ok=True)
    packets = sorted(HANDOFFS_DIR.glob("task_packet_*.md"))
    if not packets:
        print("[ERROR] No packet found in .agent/handoffs/.", file=sys.stderr)
        sys.exit(1)
    packet_path = packets[-1]
    print(f"[OK] Packet: {packet_path}")

    if args.no_launch:
        print("[INFO] --no-launch specified. Hand off the packet manually:")
        print(f'  claude "Read {packet_path.name}. Execute the mission. Do NOT use git commands."')
        return

    # 3. Launch Inner Loop
    launch_dir = worktree_dir or PROJECT_ROOT
    print(f"[Inner Loop] Launching Claude in: {launch_dir}")
    print("-" * 50)

    claude_prompt = f"Read {packet_path}. Execute the mission. Do NOT use git commands. When done, exit."
    cmd_claude = ["claude", claude_prompt]

    try:
        subprocess.check_call(cmd_claude, cwd=launch_dir)
    except subprocess.CalledProcessError:
        print("[WARN] Inner Loop exited with error code.")
    except FileNotFoundError:
        print("[ERROR] 'claude' command not found.", file=sys.stderr)
        sys.exit(1)

    print("-" * 50)
    print("[OK] Inner Loop execution complete.")
    print("[INFO] Return to Outer Loop. Run verification:")
    print(
        f"  python3 tools/orchestrator/dual_loop/verify_inner_loop_result.py "
        f"--packet {packet_path} --verbose"
    )


if __name__ == "__main__":
    main()