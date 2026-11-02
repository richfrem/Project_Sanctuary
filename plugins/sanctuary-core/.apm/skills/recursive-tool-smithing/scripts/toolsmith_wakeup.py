"""
toolsmith_wakeup.py
===================
Purpose:
    Generates a boot digest for the recursive tool-smithing loop.
    Modeled on sanctuary-guardian's guardian_wakeup.py (Protocol 128 Phase I).
    Provides: toolbox inventory, last session metrics, open gaps, system health.

Layer: Plugin Utility Script
Plugin: recursive-tool-smithing

Usage:
    python3 toolsmith_wakeup.py --mode TELEMETRY > .boot_digest.md
    python3 toolsmith_wakeup.py --bootstrap    # first-run setup
    python3 toolsmith_wakeup.py --help

Output:
    Markdown boot digest written to stdout (redirect to .boot_digest.md)

Key Functions:
    - get_toolbox_inventory(toolbox_path) -> list[dict]
    - get_last_session_metrics(results_tsv_path) -> dict
    - get_open_gaps(gaps_json_path) -> list[dict]
    - get_system_health() -> str
    - generate_boot_digest(toolbox_path, results_path, gaps_path) -> str
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


# Read all committed toolbox scripts and extract their purpose from the docstring.
def get_toolbox_inventory(toolbox_path: str = "scripts/toolbox") -> list[dict[str, Any]]:
    """Scan the toolbox directory and extract metadata from each script.

    Args:
        toolbox_path: Path to the scripts/toolbox/ directory.

    Returns:
        List of dicts with keys: name, lines, purpose (first docstring line).
    """
    p = Path(toolbox_path)
    if not p.exists():
        return []

    inventory = []
    for script in sorted(p.glob("*.py")):
        if script.name.startswith("_"):
            continue
        source = script.read_text(encoding="utf-8")
        lines = source.splitlines()
        line_count = len(lines)

        # Extract first meaningful docstring line
        purpose = "(no docstring)"
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    break
                in_docstring = True
                content = stripped.lstrip('"\'').strip()
                if content:
                    purpose = content
                    break
                continue
            if in_docstring and stripped:
                purpose = stripped
                break

        inventory.append({
            "name": script.stem,
            "lines": line_count,
            "purpose": purpose,
        })
    return inventory


# Parse the last session's metrics from results.tsv.
def get_last_session_metrics(results_tsv_path: str = "evals/results.tsv") -> dict[str, Any]:
    """Extract the most recent keep entry from the results ledger.

    Args:
        results_tsv_path: Path to the TSV results ledger.

    Returns:
        Dict with: last_commit, metric_before, metric_after, description, total_rows.
    """
    path = Path(results_tsv_path)
    if not path.exists():
        return {"status": "no_ledger"}

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) <= 1:  # header only
        return {"status": "baseline_only", "total_rows": 0}

    data_rows = [r for r in rows[1:] if r.strip()]
    keep_rows = [r for r in data_rows if "\tkeep\t" in r or r.endswith("\tkeep")]

    result: dict[str, Any] = {"total_rows": len(data_rows)}
    if keep_rows:
        last = keep_rows[-1].split("\t")
        result.update({
            "last_commit": last[0] if len(last) > 0 else "?",
            "metric_before": last[1] if len(last) > 1 else "?",
            "metric_after": last[2] if len(last) > 2 else "?",
            "description": last[-1] if last else "?",
            "status": "has_history",
        })
    else:
        result["status"] = "no_keeps_yet"
    return result


# Load unresolved gaps from the gaps registry.
def get_open_gaps(gaps_json_path: str = "gaps.json") -> list[dict[str, Any]]:
    """Load open (unresolved) gaps from the gap registry.

    Args:
        gaps_json_path: Path to gaps.json file.

    Returns:
        List of open gap dicts sorted by priority descending.
    """
    path = Path(gaps_json_path)
    if not path.exists():
        return []
    try:
        gaps = json.loads(path.read_text(encoding="utf-8"))
        open_gaps = [g for g in gaps if g.get("status") != "resolved"]
        return sorted(open_gaps, key=lambda g: g.get("priority", 0), reverse=True)
    except (json.JSONDecodeError, AttributeError):
        return []


# Get current git branch and last commit.
def get_git_state() -> dict[str, str]:
    """Return current branch name and short commit hash.

    Returns:
        Dict with: branch, commit, status (clean/dirty).
    """
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True
        ).stdout.strip()
        return {
            "branch": branch or "unknown",
            "commit": commit or "unknown",
            "status": "dirty" if dirty else "clean",
        }
    except Exception:
        return {"branch": "unknown", "commit": "unknown", "status": "unknown"}


# Generate the full boot digest markdown.
def generate_boot_digest(
    toolbox_path: str = "scripts/toolbox",
    results_path: str = "skills/recursive-tool-smithing/evals/results.tsv",
    gaps_path: str = "gaps.json",
    program_md_path: str = "program.md",
) -> str:
    """Compose the full boot digest markdown string.

    Args:
        toolbox_path: Path to toolbox directory.
        results_path: Path to results TSV.
        gaps_path: Path to gaps JSON.
        program_md_path: Path to program.md strategy file.

    Returns:
        Formatted markdown string for .boot_digest.md.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git = get_git_state()
    inventory = get_toolbox_inventory(toolbox_path)
    metrics = get_last_session_metrics(results_path)
    gaps = get_open_gaps(gaps_path)

    # System health: GREEN if clean branch on autoresearch/*, YELLOW otherwise
    health = "GREEN" if git["status"] == "clean" and git["branch"].startswith("autoresearch/") else "YELLOW"

    # Read strategy goal from program.md
    strategy_goal = "(program.md not found)"
    p = Path(program_md_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "goal" in line.lower() or "minimize" in line.lower():
                strategy_goal = line.strip().lstrip("#").strip()
                break

    lines = [
        f"# Tool-Smith Boot Digest",
        f"> Generated: {now}",
        f"> System Status: **{health}**",
        f"> Branch: `{git['branch']}` @ `{git['commit']}` ({git['status']})",
        "",
        "## Strategic Goal",
        f"{strategy_goal}",
        "",
        "## Toolbox Inventory",
        f"Total scripts: {len(inventory)}",
        "",
    ]

    if inventory:
        lines.append("| Script | Lines | Purpose |")
        lines.append("|--------|-------|---------|")
        for s in inventory:
            lines.append(f"| `{s['name']}` | {s['lines']} | {s['purpose']} |")
    else:
        lines.append("_Toolbox is empty - Bootstrap Mode required._")

    lines += [
        "",
        "## Last Session Metrics",
    ]
    if metrics.get("status") == "has_history":
        lines += [
            f"- Last commit  : `{metrics.get('last_commit', '?')}`",
            f"- Metric delta : {metrics.get('metric_before', '?')} -> {metrics.get('metric_after', '?')}",
            f"- Description  : {metrics.get('description', '?')}",
            f"- Total runs   : {metrics.get('total_rows', 0)}",
        ]
    else:
        lines.append(f"- Status: {metrics.get('status', 'unknown')}")

    lines += [
        "",
        "## Open Gaps (priority order)",
    ]
    if gaps:
        lines.append("| Priority | Slug | Description |")
        lines.append("|----------|------|-------------|")
        for g in gaps[:5]:  # show top 5
            lines.append(f"| {g.get('priority', '?')} | `{g.get('slug', '?')}` | {g.get('description', '?')} |")
    else:
        lines.append("_No open gaps registered (run gap_scanner.py to populate)._")

    lines += [
        "",
        "## Context Files",
        f"- `{program_md_path}` (strategy)",
        f"- `{toolbox_path}/` (toolbox)",
        f"- `{results_path}` (ledger)",
        f"- `{gaps_path}` (gap registry)",
        "",
        "---",
        "_Boot digest generated by toolsmith_wakeup.py_",
    ]

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for toolsmith_wakeup."""
    parser = argparse.ArgumentParser(description="Generate tool-smith boot digest.")
    parser.add_argument("--mode", choices=["TELEMETRY", "BOOTSTRAP"], default="TELEMETRY")
    parser.add_argument("--toolbox-path", default="scripts/toolbox")
    parser.add_argument("--results-path", default="skills/recursive-tool-smithing/evals/results.tsv")
    parser.add_argument("--gaps-path", default="gaps.json")
    parser.add_argument("--program-md", default="program.md")
    args = parser.parse_args()

    digest = generate_boot_digest(
        toolbox_path=args.toolbox_path,
        results_path=args.results_path,
        gaps_path=args.gaps_path,
        program_md_path=args.program_md,
    )
    print(digest)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)

    # Self-test (no CLI args)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal toolbox
        tb = Path(tmpdir) / "toolbox"
        tb.mkdir()
        (tb / "test_tool.py").write_text('"""Test tool for self-test."""\nx = 1\n')

        # Create minimal results.tsv
        results = Path(tmpdir) / "results.tsv"
        results.write_text("commit\tmetric_before\tmetric_after\tstatus\tdescription\nabc1234\t100\t80\tkeep\ttest entry\n")

        digest = generate_boot_digest(
            toolbox_path=str(tb),
            results_path=str(results),
            gaps_path="/nonexistent/gaps.json",
            program_md_path="/nonexistent/program.md",
        )

    assert "Tool-Smith Boot Digest" in digest
    assert "test_tool" in digest
    assert "abc1234" in digest
    print("toolsmith_wakeup: all self-tests passed.")
