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
        report += """
## Issues

> Manual review required. The automated check detected potential failures.
> Run a full diff review to confirm.

## Correction Prompt

> Review the FAIL items above. Fix only the specific issues identified.
> Reference the original Strategy Packet for full context.
"""

    return report, all_pass


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

    args = parser.parse_args()

    # Validate
    if not args.packet.exists():
        print(f"[ERROR] Packet not found: {args.packet}", file=sys.stderr)
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
        sys.exit(0)
    else:
        print("[RESULT] FAIL — Correction needed. See report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
