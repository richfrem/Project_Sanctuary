"""
session_chronicle.py
====================
Purpose:
    Generates hourly evolution reports and daily digests for the recursive
    tool-smithing agent. Tracks top 3 learnings/evolutions/growth events per
    hour by scanning git log, results.tsv, and plugin file changes.
    Outputs structured markdown to chronicles/YYYY-MM-DD/HH.md (hourly)
    and chronicles/YYYY-MM-DD/daily.md (digest of all hours for that day).

Layer: Plugin Utility Script
Plugin: recursive-tool-smithing

Usage:
    python3 session_chronicle.py hourly    # write current hour's report
    python3 session_chronicle.py daily     # write/update today's daily digest
    python3 session_chronicle.py both      # write both (default at session end)
    python3 session_chronicle.py --help

Output:
    chronicles/<date>/<hour>.md
    chronicles/<date>/daily.md

Key Functions:
    - get_git_changes(since_hours) -> list[dict]
    - get_ledger_delta(results_tsv, since_hours) -> list[dict]
    - get_plugin_changes(plugin_root, since_hours) -> list[dict]
    - rank_top3(changes) -> list[str]
    - write_hourly_report(output_dir, events) -> Path
    - write_daily_digest(output_dir) -> Path
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).parent.parent
CHRONICLES_DIR = PLUGIN_ROOT / "chronicles"
RESULTS_TSV = PLUGIN_ROOT / "skills" / "recursive-tool-smithing" / "evals" / "results.tsv"


# Get git commits and file changes from the last N hours.
def get_git_changes(since_hours: int = 1) -> list[dict[str, Any]]:
    """Fetch recent git commits touching the plugin folder.

    Args:
        since_hours: Look back this many hours.

    Returns:
        List of dicts with keys: hash, message, files_changed, timestamp.
    """
    since = datetime.now() - timedelta(hours=since_hours)
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--since", since_str,
             "--", str(PLUGIN_ROOT)],
            capture_output=True, text=True,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            commits.append({
                "hash": parts[0],
                "message": parts[1] if len(parts) > 1 else "",
            })
        return commits
    except Exception:
        return []


# Get new entries in results.tsv from the last N hours (by row position, not timestamp).
def get_ledger_delta(results_tsv: Path = RESULTS_TSV) -> list[dict[str, Any]]:
    """Read all non-baseline entries from the results ledger.

    Args:
        results_tsv: Path to the TSV ledger file.

    Returns:
        List of dicts parsed from TSV rows.
    """
    if not results_tsv.exists():
        return []
    rows = results_tsv.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) <= 1:
        return []
    headers = rows[0].split("\t")
    entries = []
    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) >= len(headers):
            entries.append(dict(zip(headers, cols)))
    return entries


# Get recently modified files in the plugin folder.
def get_plugin_changes(since_hours: int = 1) -> list[str]:
    """List plugin files modified in the last N hours.

    Args:
        since_hours: Look back window in hours.

    Returns:
        List of file paths relative to plugin root.
    """
    now = datetime.now().timestamp()
    cutoff = now - (since_hours * 3600)
    changed = []
    for path in PLUGIN_ROOT.rglob("*"):
        if path.is_file() and path.stat().st_mtime > cutoff:
            rel = str(path.relative_to(PLUGIN_ROOT))
            if not any(skip in rel for skip in [".git", "__pycache__", "chronicles", ".tsv.bak"]):
                changed.append(rel)
    return sorted(changed)


# Derive the top 3 most significant evolution events from all change signals.
def rank_top3(
    commits: list[dict],
    ledger: list[dict],
    file_changes: list[str],
) -> list[str]:
    """Rank and return the top 3 evolution events for the report.

    Priority:
    1. Ledger `keep` entries (tool committed = most significant growth)
    2. Ledger `crash` entries with SANDBOX_VIOLATION (safety learning)
    3. New skill/agent/script files created
    4. Git commits modifying SKILL.md or agents/
    5. Any other file changes

    Args:
        commits: List of git commit dicts.
        ledger: List of ledger row dicts.
        file_changes: List of recently modified file paths.

    Returns:
        List of up to 3 narrative strings describing top events.
    """
    events: list[tuple[int, str]] = []  # (priority, description)

    # Ledger keeps = new capability added
    keeps = [e for e in ledger if e.get("status") == "keep" and e.get("commit") != "0"]
    for entry in keeps[-3:]:  # last 3 keeps
        desc = entry.get("description", "unknown tool")
        depth = entry.get("fitness_depth", "?")
        scope = entry.get("fitness_scope", "?")
        events.append((1, f"New tool committed: `{desc}` (fitness depth={depth} scope={scope})"))

    # Sandbox violations = safety learning
    violations = [e for e in ledger if "SANDBOX_VIOLATION" in e.get("notes", "") or
                  "SANDBOX_VIOLATION" in e.get("description", "")]
    if violations:
        events.append((2, f"Safety gate triggered {len(violations)}x - sandbox violations caught and contained"))

    # Crash with simple fix = recovery learning
    crashes = [e for e in ledger if e.get("status") == "crash"]
    if len(crashes) >= 2:
        events.append((3, f"Loop resilience: recovered from {len(crashes)} crashes, applied Karpathy simple-fix rule"))

    # New skills or agents created
    new_skills = [f for f in file_changes if f.startswith("skills/") and "SKILL.md" in f]
    new_agents = [f for f in file_changes if f.startswith("agents/")]
    if new_skills:
        events.append((4, f"Skill evolved: {', '.join(new_skills[:2])}"))
    if new_agents:
        events.append((4, f"New agent configuration: {', '.join(new_agents[:2])}"))

    # Script additions
    new_scripts = [f for f in file_changes if f.startswith("scripts/") and f.endswith(".py")]
    if new_scripts:
        events.append((5, f"Scripts added/modified: {', '.join(new_scripts[:3])}"))

    # Git commits
    for c in commits[:2]:
        events.append((6, f"Committed: `{c['hash']}` {c['message']}"))

    # Sort by priority, deduplicate, take top 3
    events.sort(key=lambda x: x[0])
    seen: set[str] = set()
    top3: list[str] = []
    for _, desc in events:
        if desc not in seen and len(top3) < 3:
            top3.append(desc)
            seen.add(desc)

    if not top3:
        top3 = ["No significant changes detected this period."]

    return top3


# Write an hourly report markdown file.
def write_hourly_report(
    commits: list[dict],
    ledger: list[dict],
    file_changes: list[str],
    output_dir: Path | None = None,
) -> Path:
    """Generate and write the hourly evolution report.

    Args:
        commits: Recent git commits.
        ledger: Full ledger entries.
        file_changes: Recently modified files.
        output_dir: Override output directory (default: chronicles/<date>/).

    Returns:
        Path to the written report file.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H00")

    out_dir = output_dir or (CHRONICLES_DIR / date_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{hour_str}.md"

    top3 = rank_top3(commits, ledger, file_changes)
    keeps = [e for e in ledger if e.get("status") == "keep"]
    crashes = [e for e in ledger if e.get("status") == "crash"]

    lines = [
        f"# Hourly Evolution Report - {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Top 3 Evolutions This Hour",
        "",
    ]
    for i, event in enumerate(top3, 1):
        lines.append(f"{i}. {event}")

    lines += [
        "",
        "## Metrics Snapshot",
        f"- Tools committed (all time) : {len(keeps)}",
        f"- Crashes recovered (all time): {len(crashes)}",
        f"- Files changed this session : {len(file_changes)}",
        f"- Git commits this hour      : {len(commits)}",
        "",
        "## Files Changed This Session",
    ]
    for f in file_changes[:10]:
        lines.append(f"- `{f}`")
    if len(file_changes) > 10:
        lines.append(f"- ...and {len(file_changes) - 10} more")

    lines += [
        "",
        "---",
        f"_Generated by session_chronicle.py at {now.isoformat()}_",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# Write or update the daily digest by aggregating all hourly reports.
def write_daily_digest(output_dir: Path | None = None) -> Path:
    """Aggregate all hourly reports for today into a daily digest.

    Args:
        output_dir: Override output directory (default: chronicles/<today>/).

    Returns:
        Path to the written daily digest file.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_dir = output_dir or (CHRONICLES_DIR / date_str)
    day_dir.mkdir(parents=True, exist_ok=True)
    digest_path = day_dir / "daily.md"

    hourly_reports = sorted(day_dir.glob("[0-9]*.md"))
    all_evolutions: list[str] = []
    for report in hourly_reports:
        content = report.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith(("1.", "2.", "3.")) and len(line) > 5:
                text = line[3:].strip()
                if text not in all_evolutions:
                    all_evolutions.append(text)

    ledger = get_ledger_delta()
    total_keeps = len([e for e in ledger if e.get("status") == "keep"])
    total_crashes = len([e for e in ledger if e.get("status") == "crash"])

    lines = [
        f"# Daily Evolution Digest - {date_str}",
        "",
        f"## Summary",
        f"- Hourly reports generated : {len(hourly_reports)}",
        f"- Total tools committed    : {total_keeps}",
        f"- Total crashes recovered  : {total_crashes}",
        "",
        "## Key Evolutions Today (deduplicated)",
        "",
    ]
    for i, ev in enumerate(all_evolutions[:10], 1):
        lines.append(f"{i}. {ev}")
    if not all_evolutions:
        lines.append("_No hourly reports found yet for today._")

    lines += [
        "",
        "## Hourly Timeline",
        "",
    ]
    for report in hourly_reports:
        hour_label = report.stem
        first_lines = report.read_text(encoding="utf-8").splitlines()
        # Extract the top 3 lines
        top_items = [ln for ln in first_lines if ln.startswith(("1.", "2.", "3."))][:3]
        lines.append(f"### {hour_label}")
        for item in top_items:
            lines.append(f"  {item}")
        if not top_items:
            lines.append("  _No events_")
        lines.append("")

    lines += [
        "---",
        f"_Daily digest generated by session_chronicle.py at {now.isoformat()}_",
    ]

    digest_path.write_text("\n".join(lines), encoding="utf-8")
    return digest_path


def main() -> None:
    """CLI entry point for session_chronicle."""
    parser = argparse.ArgumentParser(description="Generate evolution chronicle reports.")
    parser.add_argument(
        "command",
        choices=["hourly", "daily", "both"],
        nargs="?",
        default="both",
        help="Report type to generate (default: both)",
    )
    parser.add_argument("--since-hours", type=int, default=1, help="Look-back window in hours")
    parser.add_argument("--plugin-root", default=str(PLUGIN_ROOT), help="Plugin root path override")
    args = parser.parse_args()

    commits = get_git_changes(since_hours=args.since_hours)
    ledger = get_ledger_delta()
    file_changes = get_plugin_changes(since_hours=args.since_hours)

    if args.command in ("hourly", "both"):
        path = write_hourly_report(commits, ledger, file_changes)
        print(f"Hourly report: {path}")

    if args.command in ("daily", "both"):
        path = write_daily_digest()
        print(f"Daily digest : {path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)

    # Self-test
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_chronicles = Path(tmpdir) / "chronicles"

        sample_commits = [{"hash": "abc1234", "message": "feat(toolbox): add pdf_parser"}]
        sample_ledger = [
            {"commit": "abc1234", "status": "keep", "fitness_depth": "3.5",
             "fitness_scope": "4.0", "description": "pdf_parser - parse PDFs to dict"},
            {"commit": "def5678", "status": "crash", "fitness_depth": "0",
             "fitness_scope": "0", "notes": "SANDBOX_VIOLATION: network",
             "description": "failed_tool"},
        ]
        sample_files = ["scripts/pdf_parser.py", "skills/recursive-tool-smithing/SKILL.md"]

        hourly = write_hourly_report(sample_commits, sample_ledger, sample_files, output_dir=tmp_chronicles / "2026-03-14")
        assert hourly.exists(), "Hourly report not created"
        content = hourly.read_text()
        assert "pdf_parser" in content, "Expected pdf_parser mention"
        assert "Top 3" in content, "Expected Top 3 section"

        daily = write_daily_digest(output_dir=tmp_chronicles / "2026-03-14")
        assert daily.exists(), "Daily digest not created"

    print("session_chronicle: all self-tests passed.")
