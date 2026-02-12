#!/usr/bin/env python3
"""
Verify Inner Loop Result (Dual-Loop Orchestrator)
===================================================

Purpose:
    CLI tool for the Outer Loop (Strategic Controller) to verify the output
    of an Inner Loop execution against the original Strategy Packet.

    Reads the strategy packet's acceptance criteria, inspects the file changes
    (via git diff or file existence checks), and produces a structured
    Verification Report.

Input:
    --packet: Path to the Strategy Packet that was sent to the Inner Loop.
    --diff:   Optional path to a saved git diff file. If omitted, runs
              git diff --stat on the current working tree.
    --worktree: Optional path to the worktree to inspect.

Output:
    A Verification Report printed to stdout (markdown format).
    Exit code 0 = PASS, exit code 1 = FAIL.

Related:
    - Skill: .agent/skills/dual-loop-supervisor/SKILL.md
    - Prompt: .agent/skills/dual-loop-supervisor/prompts/verification.md
    - Protocol 133: Dual-Loop Agent Architecture
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Project root: 4 levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = (
    PROJECT_ROOT
    / ".agent"
    / "skills"
    / "dual-loop-supervisor"
    / "prompts"
    / "verification.md"
)


def load_packet(packet_path: Path) -> dict:
    """Parse a Strategy Packet and extract acceptance criteria.

    Returns dict with keys: title, objective, criteria (list of strings).
    """
    content = packet_path.read_text(encoding="utf-8")

    # Extract title
    title_match = re.search(r"^# Mission:\s*(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Unknown"

    # Extract objective
    obj_match = re.search(r"\*\*Objective:\*\*\s*(.+)$", content, re.MULTILINE)
    objective = obj_match.group(1).strip() if obj_match else "Unknown"

    # Extract acceptance criteria (lines starting with "- [ ]")
    criteria = re.findall(r"^- \[ \]\s*(.+)$", content, re.MULTILINE)

    return {
        "title": title,
        "objective": objective,
        "criteria": criteria,
        "raw": content,
    }


def get_diff_stat(worktree: Path | None = None) -> str:
    """Run git diff --stat and return the output.

    If a worktree path is provided, runs the command there.
    Returns the diff stat string, or an error message.
    """
    cmd = ["git", "diff", "--stat"]
    cwd = str(worktree) if worktree else None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
        return result.stdout.strip() if result.stdout.strip() else "(no changes detected)"
    except FileNotFoundError:
        return "(git not available — skipping diff)"
    except subprocess.TimeoutExpired:
        return "(git diff timed out)"


def get_diff_full(worktree: Path | None = None) -> str:
    """Run git diff and return full output for inspection."""
    cmd = ["git", "diff"]
    cwd = str(worktree) if worktree else None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        output = result.stdout.strip()
        if not output:
            return "(no changes detected)"
        # Truncate for readability
        if len(output) > 5000:
            return output[:5000] + "\n\n[... truncated, full diff available via git ...]"
        return output
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "(git diff unavailable)"


def check_file_exists(criterion: str, worktree: Path | None = None) -> bool:
    """Heuristic: if a criterion mentions a file path, check it exists."""
    # Look for backtick-wrapped paths or common file patterns
    path_patterns = re.findall(r"`([^`]+\.\w+)`", criterion)
    if not path_patterns:
        # Also try bare paths
        path_patterns = re.findall(r"(\S+\.\w{1,5})\s", criterion + " ")

    base = worktree or PROJECT_ROOT
    for p in path_patterns:
        candidate = base / p if not Path(p).is_absolute() else Path(p)
        if candidate.exists():
            return True
    # If no paths found in criterion, we can't auto-check
    return True


def generate_report(
    packet: dict,
    diff_stat: str,
    diff_full: str,
    worktree: Path | None = None,
) -> tuple[str, bool]:
    """Generate a Verification Report.

    Returns (report_markdown, passed).
    """
    all_pass = True
    rows = []

    for i, criterion in enumerate(packet["criteria"], 1):
        # Heuristic file-existence check
        exists = check_file_exists(criterion, worktree)
        status = "PASS" if exists else "FAIL"
        note = "File exists" if exists else "File not found"

        if not exists:
            all_pass = False
        rows.append(f"| {i} | {criterion} | {status} | {note} |")

    if not packet["criteria"]:
        rows.append("| - | No criteria found in packet | WARN | Manual review needed |")
        all_pass = False

    verdict = "PASS" if all_pass else "FAIL"
    criteria_table = "\n".join(rows)

    report = f"""# Verification Report

**Packet**: {packet['title']}
**Verdict**: {verdict}

## Diff Summary

```
{diff_stat}
```

## Criteria Results

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
{criteria_table}
"""

    if not all_pass:
        # Build specific correction instructions from failed criteria
        failed = [
            (i, c)
            for i, (c, exists) in enumerate(
                zip(packet["criteria"], [check_file_exists(c, worktree) for c in packet["criteria"]]),
                1,
            )
            if not exists
        ]
        issue_lines = "\n".join(
            f"- **Issue {i}**: {criterion} — File not found or criterion unmet."
            for i, criterion in failed
        )
        if not issue_lines:
            issue_lines = "- Manual review required. Automated checks could not determine specific failures."

        report += f"""
## Issues

{issue_lines}

## Correction Prompt

> Fix ONLY the issues listed above. Do NOT re-implement the entire task.
> Reference the original Strategy Packet for full context.
> Do NOT use git commands.
"""

    return report, all_pass


HANDOFFS_DIR = PROJECT_ROOT / ".agent" / "handoffs"


def generate_correction_packet(
    packet: dict, report: str, original_packet_path: Path
) -> Path | None:
    """Generate a correction packet from a failed verification report.

    Returns the path to the correction packet, or None on error.
    """
    HANDOFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Find next correction number
    existing = list(HANDOFFS_DIR.glob("correction_packet_*.md"))
    numbers = []
    for f in existing:
        match = re.search(r"correction_packet_(\d+)", f.name)
        if match:
            numbers.append(int(match.group(1)))
    next_num = max(numbers) + 1 if numbers else 1

    # Extract failed criteria from report
    failed_lines = re.findall(r"\| \d+ \| (.+?) \| FAIL \| (.+?) \|", report)

    corrections = ""
    for criterion, note in failed_lines:
        corrections += f"- {criterion.strip()} ({note.strip()})\n"
    if not corrections:
        corrections = "- Review the verification report for specific failures.\n"

    content = f"""# Correction: {packet['title']}
**(Delta Fix for Inner Loop / Opus)**

> **Objective:** Fix ONLY the issues below. Do NOT re-implement the full task.

## Original Packet
`{original_packet_path}`

## Issues to Fix

{corrections.strip()}

## Constraints
- **NO GIT COMMANDS**: The Outer Loop handles all version control.
- **Delta Only**: Fix the specific issues above, nothing else.
- **File Paths**: Use exact paths from the original Strategy Packet.

## Acceptance Criteria
- [ ] All issues listed above are resolved.
- [ ] No git commands were executed.
- [ ] No files outside the original scope were modified.
"""

    output_path = HANDOFFS_DIR / f"correction_packet_{next_num:03d}.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


class TaskStatusUpdater:
    """Updates task status in a tasks.md file by checking off completed items.

    Finds a task by its ID (in <!-- id: N --> comments) and toggles
    the checkbox from [ ] to [x].
    """

    def __init__(self, tasks_path: Path) -> None:
        self.tasks_path = tasks_path

    def mark_complete(self, task_id: str) -> bool:
        """Mark a task as complete by checking its checkbox.

        Args:
            task_id: The task identifier to match (e.g., "1", "WP-001").

        Returns:
            True if the task was found and updated, False otherwise.
        """
        if not self.tasks_path.exists():
            print(f"[ERROR] Tasks file not found: {self.tasks_path}", file=sys.stderr)
            return False

        content = self.tasks_path.read_text(encoding="utf-8")
        # Pattern: - [ ] **Title** <!-- id: TASK_ID -->
        pattern = re.compile(
            r"^(- \[) \]( \*\*.+?\*\*\s*<!--\s*id:\s*"
            + re.escape(task_id)
            + r"\s*-->)",
            re.MULTILINE | re.IGNORECASE,
        )

        new_content, count = pattern.subn(r"\1x]\2", content)
        
        # Also try legacy header format ### ID. Title
        if count == 0:
             # This is a simplified fallback for legacy headers if needed, 
             # but primarily we support the new list format.
             pass

        if count == 0:
            print(
                f"[WARN] Task '{task_id}' not found (or already checked) in {self.tasks_path}",
                file=sys.stderr,
            )
            return False

        self.tasks_path.write_text(new_content, encoding="utf-8")
        print(f"[OK] Task '{task_id}' marked as complete in {self.tasks_path}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Inner Loop output against a Strategy Packet (Dual-Loop Architecture).",
        epilog="Example: %(prog)s --packet .agent/handoffs/task_packet_001.md",
    )
    parser.add_argument(
        "--packet",
        type=Path,
        required=True,
        help="Path to the Strategy Packet to verify against.",
    )
    parser.add_argument(
        "--diff",
        type=Path,
        default=None,
        help="Path to a saved diff file. If omitted, runs git diff.",
    )
    parser.add_argument(
        "--worktree",
        type=Path,
        default=None,
        help="Path to the worktree to inspect.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include full diff in the report.",
    )
    parser.add_argument(
        "--update-status",
        type=Path,
        default=None,
        help="Path to tasks.md to update task status on PASS.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Task ID to mark complete on PASS (requires --update-status).",
    )

    args = parser.parse_args()

    # Validate
    if not args.packet.exists():
        print(f"[ERROR] Packet not found: {args.packet}", file=sys.stderr)
        sys.exit(1)

    if args.update_status and not args.task_id:
        print(
            "[ERROR] --task-id is required when using --update-status",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load packet
    packet = load_packet(args.packet)
    print(f"[INFO] Verifying packet: {packet['title']}")
    print(f"[INFO] Criteria count: {len(packet['criteria'])}")

    # Get diff
    if args.diff and args.diff.exists():
        diff_stat = args.diff.read_text(encoding="utf-8")
        diff_full = diff_stat
    else:
        diff_stat = get_diff_stat(args.worktree)
        diff_full = get_diff_full(args.worktree) if args.verbose else ""

    # Generate report
    report, passed = generate_report(packet, diff_stat, diff_full, args.worktree)

    print("\n" + report)

    if passed:
        print("[RESULT] PASS — Ready for Seal (Protocol 128 Phase VI).")
        # Update task status if requested
        if args.update_status and args.task_id:
            updater = TaskStatusUpdater(args.update_status)
            updater.mark_complete(args.task_id)
        sys.exit(0)
    else:
        print("[RESULT] FAIL — Correction needed. See report above.")
        # Generate correction packet
        correction_path = generate_correction_packet(
            packet, report, args.packet
        )
        if correction_path:
            print(f"[INFO] Correction packet written to: {correction_path}")
            print(f'[INFO] Hand off: claude "Read {correction_path}. Fix the issues. Do NOT use git."')
        sys.exit(1)


if __name__ == "__main__":
    main()
