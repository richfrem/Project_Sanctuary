"""
gap_scanner.py
==============
Purpose:
    Scans a task log or capability-gap registry and returns a priority-sorted list
    of unresolved capability gaps for the recursive tool-smithing evolution loop.

Layer: Plugin Utility Script
Plugin: recursive-tool-smithing

Usage:
    python3 gap_scanner.py --toolbox-path scripts/toolbox --gap-log gaps.json
    python3 gap_scanner.py --help

Input Files:
    - gaps.json (or inline gap registry)
    - scripts/toolbox/ (scans for existing tools to avoid duplicates)

Output:
    - JSON list of gaps sorted by priority (highest first)

Key Functions:
    - scan_toolbox(toolbox_path) -> list[str]
    - load_gaps(gap_log_path) -> list[dict]
    - filter_resolved(gaps, existing_tools) -> list[dict]
    - sort_by_priority(gaps) -> list[dict]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


# Scan the toolbox directory for already-committed scripts.
def scan_toolbox(toolbox_path: str) -> list[str]:
    """Return a list of script slugs already present in the toolbox.

    Args:
        toolbox_path: Path to the scripts/toolbox/ directory.

    Returns:
        List of script stem names (e.g. ['get_stock_alpha', 'clean_memory_logs']).
    """
    p = Path(toolbox_path)
    if not p.exists():
        return []
    return [f.stem for f in p.glob("*.py") if not f.name.startswith("_")]


# Load the gap registry from a JSON file.
def load_gaps(gap_log_path: str) -> list[dict[str, Any]]:
    """Load gaps from a JSON registry file.

    Args:
        gap_log_path: Path to the gaps.json file.

    Returns:
        List of gap dicts with keys: slug, description, priority, status.

    Raises:
        FileNotFoundError: If the gap log does not exist.
        ValueError: If the JSON is malformed.
    """
    path = Path(gap_log_path)
    if not path.exists():
        raise FileNotFoundError(f"Gap log not found: {gap_log_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Gap log must be a JSON array of gap objects.")
    return data


# Remove gaps that already have a script in the toolbox.
def filter_resolved(
    gaps: list[dict[str, Any]], existing_tools: list[str]
) -> list[dict[str, Any]]:
    """Remove gaps whose slug already exists as a committed tool.

    Args:
        gaps: Raw list of gap dicts.
        existing_tools: List of already-committed script stems.

    Returns:
        Filtered list of unresolved gaps.
    """
    tool_set = set(existing_tools)
    return [g for g in gaps if g.get("slug") not in tool_set and g.get("status") != "resolved"]


# Sort gaps descending by priority (1 = lowest, 5 = highest).
def sort_by_priority(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort gaps by priority descending.

    Args:
        gaps: List of gap dicts with a 'priority' field (int 1-5).

    Returns:
        Sorted list, highest priority first.
    """
    return sorted(gaps, key=lambda g: g.get("priority", 0), reverse=True)


def main() -> None:
    """CLI entry point for gap_scanner."""
    parser = argparse.ArgumentParser(
        description="Scan for unresolved capability gaps in the tool-smithing loop."
    )
    parser.add_argument("--toolbox-path", default="scripts/toolbox", help="Path to toolbox dir")
    parser.add_argument("--gap-log", required=True, help="Path to gaps.json registry")
    args = parser.parse_args()

    existing = scan_toolbox(args.toolbox_path)
    gaps = load_gaps(args.gap_log)
    unresolved = filter_resolved(gaps, existing)
    ranked = sort_by_priority(unresolved)

    print(json.dumps(ranked, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)
    # Self-test
    import tempfile

    sample_gaps = [
        {"slug": "pdf_parser", "description": "Parse PDFs to structured data", "priority": 4, "status": "open"},
        {"slug": "existing_tool", "description": "Already committed", "priority": 5, "status": "open"},
        {"slug": "cache_manager", "description": "LRU cache for API responses", "priority": 3, "status": "open"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_gaps, f)
        tmp_path = f.name

    existing = ["existing_tool"]
    gaps = load_gaps(tmp_path)
    filtered = filter_resolved(gaps, existing)
    ranked = sort_by_priority(filtered)

    assert len(ranked) == 2, f"Expected 2 gaps, got {len(ranked)}"
    assert ranked[0]["slug"] == "pdf_parser", "Expected pdf_parser first (priority 4)"
    assert ranked[1]["slug"] == "cache_manager", "Expected cache_manager second (priority 3)"
    os.unlink(tmp_path)
    print("gap_scanner: all self-tests passed.")
