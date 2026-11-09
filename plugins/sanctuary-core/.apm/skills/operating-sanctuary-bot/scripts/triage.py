#!/usr/bin/env python3
"""triage.py - pure step selection for daemon campaigns (no I/O except optional manifest lookup).

Step states: PENDING -> RUNNING -> DONE | FAILED ; BLOCKED when a dependency FAILED or is BLOCKED.

Key Input Dependencies:
- Step dicts from the run state (no file I/O except optional LEARNING/missions/<id>.json for priority_score)

Key Functions:
- propagate_blocks(): Mark PENDING steps whose dependencies failed or were blocked. Returns newly blocked ids.
- mission_priority(): Sanctuary priority_score of a linked mission manifest, if one exists.
- score(): A step's priority: its own `priority`, else its mission's priority_score, else 0.
- actionable(): PENDING steps whose dependencies are all DONE.
- next_step(): pipeline: first actionable step in declared order. backlog: highest score (ties: declared order).
- outcome(): COMPLETED if all DONE; PARTIAL if all terminal with some FAILED/BLOCKED; STALLED otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

TERMINAL = {"DONE", "FAILED", "BLOCKED", "SKIPPED"}


# Mark PENDING steps whose dependencies failed or were blocked.
def propagate_blocks(steps: list[dict]) -> list[str]:
    """Mark PENDING steps whose dependencies failed or were blocked. Returns newly blocked ids."""
    by_id = {s["id"]: s for s in steps}
    newly, changed = [], True
    while changed:
        changed = False
        for s in steps:
            if s.get("status", "PENDING") != "PENDING":
                continue
            skipped = [d for d in s.get("depends_on", []) if by_id.get(d, {}).get("status") == "SKIPPED"]
            if skipped:
                s["status"], s["reason"] = "SKIPPED", f"dependency {', '.join(skipped)} skipped"
                newly.append(s["id"])
                changed = True
                continue
            bad = [d for d in s.get("depends_on", []) if by_id.get(d, {}).get("status") in ("FAILED", "BLOCKED")]
            missing = [d for d in s.get("depends_on", []) if d not in by_id]
            if bad or missing:
                s["status"] = "BLOCKED"
                s["reason"] = f"dependency {', '.join(bad + missing)} {'failed or blocked' if bad else 'unknown'}"
                newly.append(s["id"])
                changed = True
    return newly


# Sanctuary priority_score of a linked mission manifest, if one exists.
def mission_priority(mission_id: str | None, repo_root: Path | None) -> float | None:
    """Sanctuary priority_score of a linked mission manifest, if one exists."""
    if not mission_id or not repo_root:
        return None
    p = repo_root / "LEARNING" / "missions" / f"{mission_id}.json"
    try:
        return float(json.loads(p.read_text()).get("priority_score"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


# A step's priority.
def score(step: dict, repo_root: Path | None = None) -> float:
    """A step's priority: its own `priority`, else its mission's priority_score, else 0."""
    if step.get("priority") is not None:
        return float(step["priority"])
    mp = mission_priority(step.get("mission_id"), repo_root)
    return mp if mp is not None else 0.0


# PENDING steps whose dependencies are all DONE.
def actionable(steps: list[dict]) -> list[dict]:
    """PENDING steps whose dependencies are all DONE."""
    by_id = {s["id"]: s for s in steps}
    return [s for s in steps if s.get("status", "PENDING") == "PENDING"
            and all(by_id.get(d, {}).get("status") == "DONE" for d in s.get("depends_on", []))]


# pipeline.
def next_step(steps: list[dict], mode: str, repo_root: Path | None = None) -> dict | None:
    """pipeline: first actionable step in declared order. backlog: highest score (ties: declared order)."""
    propagate_blocks(steps)
    ready = actionable(steps)
    if not ready:
        return None
    if mode == "backlog":
        order = {s["id"]: i for i, s in enumerate(steps)}
        return max(ready, key=lambda s: (score(s, repo_root), -order[s["id"]]))
    return ready[0]


# COMPLETED if all DONE; PARTIAL if all terminal with some FAILED/BLOCKED; STALLED otherwise.
def outcome(steps: list[dict]) -> str:
    """COMPLETED if all DONE; PARTIAL if all terminal with some FAILED/BLOCKED; STALLED otherwise."""
    states = {s.get("status", "PENDING") for s in steps}
    if states <= {"DONE", "SKIPPED"}:
        return "COMPLETED"
    if states <= TERMINAL:
        return "PARTIAL"
    return "STALLED"
