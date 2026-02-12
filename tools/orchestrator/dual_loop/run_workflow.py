#!/usr/bin/env python3
"""
Dual-Loop Wrapper CLI (Python Port of spec_kitty_dual_loop_wrapper.sh)
Orchestrates the Dual-Loop Workflow:
1. Create Worktree (Spec Kitty Implement)
2. Generate Strategy Packet
3. Launch Inner Loop (Claude) in isolation
4. (Optional) Run Verification

Usage:
  python3 spec_kitty_dual_loop_wrapper.py <TASK_ID> [--tasks-file PATH]
"""

import argparse
import os
import re
import sys
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate Dual-Loop Workflow")
    parser.add_argument("task_id", help="Task ID (e.g., '3', 'WP-06')")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default="kitty-specs/001-dual-loop-agent-architecture/tasks.md",
        help="Path to tasks.md relative to project root",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    print(f"🚀 [Outer Loop] Initiating Dual-Loop Sequence for Task {args.task_id}...")

    # 1. Create Worktree (Phase II)
    print("📂 [Spec Kitty] Creating isolated worktree...")
    cmd_implement = [
        "spec-kitty",
        "agent",
        "workflow",
        "implement",
        "--task-id",
        args.task_id,
        "--agent",
        "DualLoopBot",
    ]
    
    # We need to capture stdout/stderr to find the worktree path, but also show progress?
    # Spec Kitty output can be long. Let's capture it.
    try:
        result = subprocess.run(cmd_implement, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("❌ 'spec-kitty' command not found. Ensure it is in your PATH or npm/node_modules.")
        sys.exit(1)

    output = result.stdout + result.stderr
    
    if result.returncode != 0 and "SUCCESS" not in output:
        print("❌ Spec Kitty command failed.")
        print(output)
        sys.exit(1)

    # Extract worktree path: matching "cd .worktrees/..."
    match = re.search(r"cd\s+(.worktrees/[^\s]+)", output)
    if not match:
        print("❌ Could not determine worktree directory from Spec Kitty output.")
        print("Output was:\n" + output[-500:]) # Show last 500 chars
        sys.exit(1)

    worktree_rel_path = match.group(1)
    worktree_dir = project_root / worktree_rel_path
    
    if not worktree_dir.exists():
         print(f"❌ Worktree directory not found: {worktree_dir}")
         sys.exit(1)

    print(f"✅ Worktree created at: {worktree_rel_path}")

    # 2. Generate Strategy Packet (Phase III / Audit)
    print("📜 [Dual Loop] Generating Strategy Packet...")
    packet_name = f"strategy_packet_{args.task_id}.md"
    packet_path = worktree_dir / packet_name
    
    # We run the generator script. It should be importable or run via subprocess.
    # We use absolute path to tool.
    generator_tool = project_root / "tools/orchestrator/dual_loop/generate_strategy_packet.py"
    tasks_file_abs = project_root / args.tasks_file

    cmd_generate = [
        sys.executable,
        str(generator_tool),
        "--tasks-file",
        str(tasks_file_abs),
        "--task-id",
        args.task_id,
        "--output",
        str(packet_path),
    ]

    gen_result = subprocess.run(cmd_generate, cwd=project_root)
    if gen_result.returncode != 0:
        print("❌ Failed to generate Strategy Packet.")
        sys.exit(1)

    print(f"✅ Packet generated: {packet_path}")

    # 3. Launch Inner Loop (Phase III / Execution)
    print("🤖 [Inner Loop] Launching Claude Code...")
    print("-" * 50)
    print(f"CONTEXT: {worktree_dir}")
    print(f"PACKET:  {packet_name}")
    print("-" * 50)

    # Interactive Claude Session
    claude_prompt = f"Read {packet_name}. Execute the mission. Do NOT use git commands. When done, exit."
    cmd_claude = ["claude", claude_prompt]

    try:
        # Run interactively in the worktree
        subprocess.check_call(cmd_claude, cwd=worktree_dir)
    except subprocess.CalledProcessError:
        print("⚠️  Inner Loop exited with error code.")
    except FileNotFoundError:
        print("❌ 'claude' command not found.")
        sys.exit(1)

    print("-" * 50)
    print("✅ [Inner Loop] Execution complete.")
    print("🔙 Returning to Outer Loop controller.")

    # 4. Prompt for Verification (Phase V)
    # Basic input prompt
    try:
        choice = input("Run verification now? (y/n) ").strip().lower()
    except EOFError:
        choice = "n"

    if choice == "y":
        verifier_tool = project_root / "tools/orchestrator/dual_loop/verify_inner_loop_result.py"
        cmd_verify = [
            sys.executable,
            str(verifier_tool),
            "--packet",
            str(packet_path),
            "--worktree",
            str(worktree_dir),
            "--update-status",
            str(tasks_file_abs),
            "--task-id",
            args.task_id,
            "--verbose",
        ]
        subprocess.run(cmd_verify, cwd=project_root)
    else:
        print("⚠️  Verification skipped.")


if __name__ == "__main__":
    main()
