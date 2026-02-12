#!/usr/bin/env python3
"""
Generate Strategy Packet (Dual-Loop Orchestrator)
==================================================

Purpose:
    CLI tool for the Outer Loop (Strategic Controller) to distill a tasks.md
    item into a minimal, token-efficient Strategy Packet for the Inner Loop.

    Reads the task definition, loads the strategy generation prompt template,
    and produces a self-contained markdown packet.

Input:
    --tasks-file: Path to tasks.md
    --task-id:    Identifier of the task to extract (e.g., "WP-001" or "A")
    --output:     Optional output path (default: .agent/handoffs/)
    --spec:       Optional path to spec.md for context injection
    --plan:       Optional path to plan.md for context injection

Output:
    A Strategy Packet markdown file written to .agent/handoffs/ or stdout.

Related:
    - Skill: .agent/skills/dual-loop-supervisor/SKILL.md
    - Prompt: .agent/skills/dual-loop-supervisor/prompts/strategy_generation.md
    - Protocol 133: Dual-Loop Agent Architecture
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Project root: 4 levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HANDOFFS_DIR = PROJECT_ROOT / ".agent" / "handoffs"
PROMPT_PATH = (
    PROJECT_ROOT
    / ".agent"
    / "skills"
    / "dual-loop-supervisor"
    / "prompts"
    / "strategy_generation.md"
)


def parse_tasks_file(tasks_path: Path) -> list[dict]:
    """Parse a tasks.md file into a list of task dicts.

    Extracts tasks by looking for markdown headings that contain task
    identifiers (e.g., '### A.', '### WP-001:').

    Returns a list of dicts with keys: id, title, body.
    """
    content = tasks_path.read_text(encoding="utf-8")
    tasks = []
    # Match headings like "### A. Title" or "### WP-001: Title"
    pattern = r"^###\s+([A-Z0-9][\w-]*)[.:\s]+(.+)$"
    lines = content.split("\n")
    current_task = None

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_task:
                tasks.append(current_task)
            current_task = {
                "id": match.group(1),
                "title": match.group(2).strip(),
                "body": "",
            }
        elif current_task:
            current_task["body"] += line + "\n"

    if current_task:
        tasks.append(current_task)

    return tasks


def find_task(tasks: list[dict], task_id: str) -> dict | None:
    """Find a task by its identifier (case-insensitive)."""
    for task in tasks:
        if task["id"].lower() == task_id.lower():
            return task
    return None


def load_prompt_template() -> str:
    """Load the strategy generation prompt template."""
    if not PROMPT_PATH.exists():
        print(f"[ERROR] Prompt template not found: {PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_context_file(path: Path | None, label: str) -> str:
    """Load an optional context file (spec or plan), return excerpt or empty."""
    if path is None or not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    # Truncate to first 2000 chars for token efficiency
    if len(content) > 2000:
        content = content[:2000] + "\n\n[... truncated for token efficiency ...]"
    return f"\n### {label} (excerpt)\n```\n{content}\n```\n"


def generate_packet(
    task: dict,
    spec_excerpt: str,
    plan_excerpt: str,
    spec_path: str | None,
    plan_path: str | None,
) -> str:
    """Generate a Strategy Packet markdown string from a task definition."""
    spec_ref = spec_path or "[not provided]"
    plan_ref = plan_path or "[not provided]"

    packet = f"""# Mission: {task['title']}
**(Strategy Packet for Inner Loop / Opus)**

> **Objective:** Execute the task defined below. This packet is your entire context.

## 1. Context
- **Spec**: `{spec_ref}`
- **Plan**: `{plan_ref}`
- **Goal**: {task['title']}

## 2. Tasks

{task['body'].strip()}

## 3. Constraints
- **NO GIT COMMANDS**: The Outer Loop handles all version control.
- **Token Efficiency**: Produce only the requested artifacts, nothing extra.
- **File Paths**: Use exact paths as specified in the task.

## 4. Acceptance Criteria
- [ ] All files specified in section 2 exist and are correctly implemented.
- [ ] No git commands were executed.
- [ ] Code follows project coding conventions.
"""
    return packet


def next_packet_number() -> int:
    """Determine the next packet number from existing handoff files."""
    HANDOFFS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(HANDOFFS_DIR.glob("task_packet_*.md"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        match = re.search(r"task_packet_(\d+)", f.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Strategy Packet for the Inner Loop (Dual-Loop Architecture).",
        epilog="Example: %(prog)s --tasks-file specs/001/tasks.md --task-id A",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        required=True,
        help="Path to the tasks.md file.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Task identifier to extract (e.g., 'A', 'WP-001').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Default: .agent/handoffs/task_packet_NNN.md",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional path to spec.md for context injection.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Optional path to plan.md for context injection.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print packet to stdout instead of writing to file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing.",
    )

    args = parser.parse_args()

    # Validate input
    if not args.tasks_file.exists():
        print(f"[ERROR] Tasks file not found: {args.tasks_file}", file=sys.stderr)
        sys.exit(1)

    # Parse and find task
    tasks = parse_tasks_file(args.tasks_file)
    if not tasks:
        print(f"[ERROR] No tasks found in {args.tasks_file}", file=sys.stderr)
        sys.exit(1)

    task = find_task(tasks, args.task_id)
    if not task:
        available = ", ".join(t["id"] for t in tasks)
        print(
            f"[ERROR] Task '{args.task_id}' not found. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load optional context
    spec_excerpt = load_context_file(args.spec, "Spec")
    plan_excerpt = load_context_file(args.plan, "Plan")

    # Generate packet
    packet = generate_packet(
        task,
        spec_excerpt,
        plan_excerpt,
        str(args.spec) if args.spec else None,
        str(args.plan) if args.plan else None,
    )

    if args.dry_run:
        print("[DRY RUN] Would generate packet:")
        print(packet)
        return

    if args.stdout:
        print(packet)
        return

    # Write to file
    output_path = args.output or (
        HANDOFFS_DIR / f"task_packet_{next_packet_number():03d}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet, encoding="utf-8")
    print(f"[OK] Strategy Packet written to: {output_path}")


if __name__ == "__main__":
    main()
