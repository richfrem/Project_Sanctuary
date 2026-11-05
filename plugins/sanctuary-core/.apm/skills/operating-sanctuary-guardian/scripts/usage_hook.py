#!/usr/bin/env python3
"""usage_hook.py - Flight-board hook: measured usage at mission checkout/checkin (MISSION-035).

Purpose:
    Gather real usage telemetry as a by-product of normal mission work: checkout stores a zero-token
    usage snapshot on the lease; checkin takes a second one, computes window deltas, wall-clock time,
    and (when known) measured tokens, and records them. Best-effort by design: a failing probe never
    blocks a checkout or checkin. Disable with --no-usage or SANCTUARY_NO_USAGE=1.
Layer: sanctuary-core / operating-sanctuary-guardian (uses evaluating-agent-work/scripts/usage_probe.py).
Function Index: backend_for_agent, start_snapshot, claude_project_dirs, finish
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_SKILLS = Path(__file__).resolve().parents[2]
_PROBE = _SKILLS / "evaluating-agent-work" / "scripts" / "usage_probe.py"
_BUDGET = _SKILLS / "evaluating-agent-work" / "scripts" / "track_budget.py"


def _probe():
    if not _PROBE.exists():
        return None
    spec = importlib.util.spec_from_file_location("usage_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def backend_for_agent(agent: str) -> Optional[str]:
    a = agent.lower()
    if "claude" in a:
        return "claude"
    if "antigravity" in a or a == "agy":
        return "agy"
    if "codex" in a or "astra" in a:
        return "codex"
    if "mlx" in a or "qwen" in a or "local" in a:
        return "local"
    return None


def _disabled(no_usage: bool) -> bool:
    return no_usage or os.environ.get("SANCTUARY_NO_USAGE") == "1"


def start_snapshot(no_usage: bool = False, snapshot_fn: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """Zero-token snapshot stored on the lease; None when disabled or the probe fails."""
    if _disabled(no_usage):
        return None
    try:
        if snapshot_fn is None:
            mod = _probe()
            if mod is None:
                return None
            snapshot_fn = mod.snapshot
        return snapshot_fn(["claude", "agy", "codex"])
    except Exception:  # noqa: BLE001 - never block a checkout
        return None


def headroom_warnings(snap: Optional[Dict[str, Any]], agent: str, floor: float = 15.0) -> List[Dict[str, Any]]:
    """Backends whose tightest 5-hour/weekly window is under the floor; the agent's own backend is flagged."""
    if not snap:
        return []
    mod = _probe()
    if mod is None:
        return []
    own = backend_for_agent(agent)
    return [{**r, "own_backend": r["backend"] == own} for r in mod.headroom_report(snap, floor) if r["constrained"]]


LATEST_REL = Path(".agent") / "learning" / "usage_latest.json"  # gitignored cache read by the boot briefing
LATEST_MAX_AGE_S = 6 * 3600


def save_latest(root: Path, snap: Optional[Dict[str, Any]]) -> None:
    """Cache the newest snapshot so the boot briefing can show the fuel gauge without calling any CLI."""
    if not snap:
        return
    try:
        p = root / LATEST_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap), encoding="utf-8")
    except OSError:
        pass


def load_latest(root: Path, max_age_s: int = LATEST_MAX_AGE_S, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Cached snapshot if fresh enough (else None)."""
    p = root / LATEST_REL
    try:
        if (now or time.time()) - p.stat().st_mtime > max_age_s:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def claude_project_dirs(root: Path, projects_dir: Optional[Path] = None) -> List[Path]:
    """Claude transcript dirs for this repo and its worktrees (encoded-path prefix match)."""
    projects_dir = projects_dir or Path.home() / ".claude" / "projects"
    enc = re.sub(r"[^A-Za-z0-9]", "-", str(root))
    if not projects_dir.exists():
        return []
    return [d for d in projects_dir.iterdir() if d.is_dir() and d.name.startswith(enc)]


def finish(root: Path, lease: Dict[str, Any], mission: str, agent: str, task_class: str = "mission",
           tokens: Optional[int] = None, quality: Optional[int] = None, retries: int = 0,
           no_usage: bool = False, snapshot_fn: Optional[Callable] = None,
           projects_dir: Optional[Path] = None, record: bool = True, model: Optional[str] = None,
           codex_sessions_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Second snapshot, deltas, wall-clock and tokens; records a measured budget entry when tokens are known."""
    if _disabled(no_usage) or not lease:
        return None
    try:
        mod = _probe()
        if mod is None:
            return None
        snapshot_fn = snapshot_fn or mod.snapshot
        before = lease.get("usage_start")
        after = snapshot_fn(["claude", "agy", "codex"])
        save_latest(root, after)
        started = lease.get("checked_out_timestamp") or time.time()
        wall = int(time.time() - started)
        summary: Dict[str, Any] = {"task_class": task_class, "wall_seconds": wall, "retries": retries,
                                   "backend": backend_for_agent(agent), "tokens_source": None}
        if quality is not None:
            summary["quality"] = quality
        if before:
            summary["window_delta"] = mod.diff(before, after)
        concurrent = False
        backend = summary["backend"]
        if tokens is not None:
            summary.update(tokens=int(tokens), tokens_source="vendor_reported")
        elif backend == "claude":
            start_iso = datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            tot = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                   "cache_write_1h_tokens": 0}
            for d in claude_project_dirs(root, projects_dir):
                r = mod.claude_tokens_between(d, start_iso, end_iso)
                concurrent = concurrent or r["concurrent"]
                for k in tot:
                    tot[k] += r[k]
            summary.update(tokens=tot["input_tokens"] + tot["output_tokens"] + tot["cache_creation_input_tokens"],
                           tokens_in=tot["input_tokens"], tokens_out=tot["output_tokens"],
                           tokens_cache_write=tot["cache_creation_input_tokens"] - tot["cache_write_1h_tokens"],
                           tokens_cache_write_1h=tot["cache_write_1h_tokens"],
                           tokens_cache_read=tot["cache_read_input_tokens"], tokens_source="claude_transcripts")
        elif backend == "codex":
            start_iso = datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = mod.codex_tokens_between(codex_sessions_dir, start_iso, end_iso)
            concurrent = r["concurrent"]
            if r["total_tokens"]:
                cached = r.get("cached_input_tokens", 0)
                summary.update(tokens=r["total_tokens"], tokens_source="codex_rollouts",
                               tokens_in=max(0, r.get("input_tokens", 0) - cached), tokens_out=r.get("output_tokens", 0),
                               tokens_cache_read=cached)
        summary["concurrent"] = concurrent
        if model:
            summary["model"] = model
            cost = mod.cost_usd(model, summary.get("tokens_in", summary.get("tokens", 0)) or 0, summary.get("tokens_out", 0) or 0,
                                cache_read=summary.get("tokens_cache_read", 0), cache_write=summary.get("tokens_cache_write", 0),
                                cache_write_1h=summary.get("tokens_cache_write_1h", 0))
            summary["cost_usd"] = cost  # None when the model is not in the cli-agent catalog (never guessed)
        mod.assert_no_secrets(summary)
        if record and summary.get("tokens"):
            cmd = [sys.executable, str(_BUDGET), "record", "--task-id", mission, "--backend", backend or "unknown",
                   "--tokens-est", str(summary["tokens"]), "--source", "measured", "--task-class", task_class,
                   "--outcome", "completed", "--wall-seconds", str(wall), "--retries", str(retries)]
            if summary.get("tokens_in") is not None:
                cmd += ["--tokens-in", str(summary["tokens_in"]), "--tokens-out", str(summary["tokens_out"]),
                        "--tokens-cache-read", str(summary.get("tokens_cache_read", 0)),
                        "--tokens-cache-write", str(summary.get("tokens_cache_write", 0)),
                        "--tokens-cache-write-1h", str(summary.get("tokens_cache_write_1h", 0))]
            if quality is not None:
                cmd += ["--quality", str(quality)]
            if model:
                cmd += ["--model", model]
            if summary.get("cost_usd") is not None:
                cmd += ["--cost-usd", str(summary["cost_usd"])]
            if concurrent:
                cmd.append("--concurrent")
            if summary.get("window_delta"):
                cmd += ["--window-delta", json.dumps(summary["window_delta"])]
            subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(root))
        return summary
    except Exception as exc:  # noqa: BLE001 - never block a checkin
        return {"error": type(exc).__name__}
