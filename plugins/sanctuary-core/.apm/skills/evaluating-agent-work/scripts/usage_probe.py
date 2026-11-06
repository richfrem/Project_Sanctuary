#!/usr/bin/env python3
"""usage_probe.py - Measured agent usage telemetry (MISSION-035).

Purpose:
    Deterministic, orchestrator-run before/after usage snapshots across the model backends
    (claude, agy, codex, local) so token and quota-window consumption is measured, not estimated
    or self-reported by the worker model. Feeds track_budget.py (source=measured) and a
    per-task-class cost predictor / backend recommender.
Layer: sanctuary-core / evaluating-agent-work. Measurement (parsers, snapshot/diff, secret scrub, prices) lives in the shared
    evaluating-agent-outcomes/scripts/usage_measure.py; this file keeps Sanctuary policy: floors, static ceilings, the (falsified,
    MISSION-035) per-class predictor, headroom/recommend and the CLI that records BUDGET_CONSUMED.
Key inputs: `claude -p /usage`, `agy -p /usage --output-format json` (both verified 0-token),
    ~/.codex/sessions rollout files (token_count events), ~/.claude/projects transcripts.
Key outputs: snapshot/diff JSON; BUDGET_CONSUMED events (via track_budget.py); recommendations.
Safety: reads named usage fields only; never opens auth files; every emitted record passes
    assert_no_secrets(). Probes never send a prompt to a model (a prompt would spend tokens).
Usage:
    python3 usage_probe.py snapshot [--backends claude,agy,codex] [--out FILE]
    python3 usage_probe.py diff BEFORE.json AFTER.json
    python3 usage_probe.py headroom [--floor 15]
    python3 usage_probe.py recommend --task-class CLASS [--floor 15]
    python3 usage_probe.py calibrate [--min-pct 5]
    python3 usage_probe.py evaluate [--min-train 8]
Function Index:
    parse_claude_usage, parse_agy_usage, reported_tokens, read_codex_latest, claude_tokens_between,
    snapshot, diff, assert_no_secrets, load_measured, fit_predictor, predict, evaluate_predictor,
    recommend, main
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared  # noqa: E402,F401  (locates the shared evaluating-agent-outcomes skill and puts it on sys.path)
# Generic measurement is single-sourced in the companion skill (MISSION-055); names are re-exported for existing callers/tests.
from usage_measure import (SECRET_PATTERNS, CATALOG_DIRS, assert_no_secrets, parse_claude_usage, parse_agy_usage,  # noqa: E402,F401
                           reported_tokens, read_codex_latest, _parse_ts, claude_tokens_between, parse_agy_json_usage,
                           codex_tokens_between, _run, snapshot, diff, find_catalog_dir, load_prices, cost_usd)

STATIC_CEILINGS = {"claude": 60000, "codex": 80000, "agy": 150000, "local": 80000}
DEFAULT_FLOOR_PCT = 15.0


def get_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()




# --------------------------------------------------------------------------- parsers
















# --------------------------------------------------------------------------- snapshot / diff






# --------------------------------------------------------------------------- predictor / recommender
def load_measured(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Measured BUDGET_CONSUMED events (source=measured) from events.jsonl."""
    root = root or get_project_root()
    path = root / ".agent" / "learning" / "events.jsonl"
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("event_type") == "BUDGET_CONSUMED" and rec.get("source") == "measured" and rec.get("task_class"):
            out.append(rec)
    return out


def fit_predictor(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Median tokens per (backend, task_class), with per-backend fallback medians."""
    by_cls: Dict[str, List[int]] = {}
    by_backend: Dict[str, List[int]] = {}
    for r in records:
        by_cls.setdefault(f"{r['backend']}|{r['task_class']}", []).append(int(r["tokens_est"]))
        by_backend.setdefault(r["backend"], []).append(int(r["tokens_est"]))
    return {"class": {k: statistics.median(v) for k, v in by_cls.items()},
            "class_n": {k: len(v) for k, v in by_cls.items()},
            "backend": {k: statistics.median(v) for k, v in by_backend.items()}}


def predict(model: Dict[str, Any], backend: str, task_class: str) -> Dict[str, Any]:
    key = f"{backend}|{task_class}"
    if key in model["class"]:
        return {"tokens": model["class"][key], "basis": "class_median", "n": model["class_n"][key]}
    if backend in model["backend"]:
        return {"tokens": model["backend"][backend], "basis": "backend_median", "n": 0}
    return {"tokens": STATIC_CEILINGS.get(backend, 80000), "basis": "static_ceiling", "n": 0}


def _mape(pairs: List[tuple]) -> Optional[float]:
    vals = [abs(p - a) / a for p, a in pairs if a]
    return round(100.0 * sum(vals) / len(vals), 2) if vals else None


def evaluate_predictor(records: List[Dict[str, Any]], min_train: int = 8) -> Dict[str, Any]:
    """Chronological train/held-out split: MAPE of the class-median predictor vs the static ceiling."""
    recs = sorted(records, key=lambda r: r.get("timestamp", ""))
    if len(recs) < min_train + 2:
        return {"status": "insufficient_data", "n": len(recs), "needed": min_train + 2}
    cut = max(min_train, int(len(recs) * 0.7))
    cut = min(cut, len(recs) - 1)
    train, held = recs[:cut], recs[cut:]
    model = fit_predictor(train)
    learned = _mape([(predict(model, r["backend"], r["task_class"])["tokens"], int(r["tokens_est"])) for r in held])
    static = _mape([(STATIC_CEILINGS.get(r["backend"], 80000), int(r["tokens_est"])) for r in held])
    improvement = None if learned is None or not static else round(100.0 * (static - learned) / static, 2)
    return {"status": "ok", "n_train": len(train), "n_held_out": len(held), "mape_learned_pct": learned,
            "mape_static_ceiling_pct": static, "improvement_pct": improvement}


def _remaining_pct(snap_backend: Dict[str, Any], backend: str) -> Optional[float]:
    """Smallest remaining headroom (percent) across a backend's windows; None if unknown."""
    vals: List[float] = []
    if backend == "claude":
        for k in ("session_pct_used", "week_pct_used"):
            if snap_backend.get(k) is not None:
                vals.append(100.0 - snap_backend[k])
    elif backend == "agy":
        vals = [r["remaining_pct"] for r in snap_backend.get("buckets", [])]
    elif backend == "codex":
        for k in ("five_hour_pct_used", "weekly_pct_used"):
            if snap_backend.get(k) is not None:
                vals.append(100.0 - snap_backend[k])
    return min(vals) if vals else None


def recommend(records: List[Dict[str, Any]], snap: Dict[str, Any], task_class: str,
              floor: float = DEFAULT_FLOOR_PCT) -> List[Dict[str, Any]]:
    """Rank backends for a task class: unconstrained first, then lowest expected tokens.

    Advisory only: never overrides the human or the fallback-chain rules. A backend whose
    tightest window has less than `floor` percent remaining is marked constrained.
    """
    model = fit_predictor(records)
    out = []
    for backend in ("claude", "codex", "agy", "local"):
        pred = predict(model, backend, task_class)
        head = _remaining_pct(snap.get("backends", {}).get(backend, {}), backend) if backend != "local" else None
        constrained = head is not None and head < floor
        conf = "high" if pred["n"] >= 5 else "medium" if pred["n"] >= 2 else "low"
        out.append({"backend": backend, "expected_tokens": pred["tokens"], "basis": pred["basis"],
                    "evidence_n": pred["n"], "confidence": conf, "headroom_pct": None if head is None else round(head, 1),
                    "constrained": constrained})
    out.sort(key=lambda r: (r["constrained"], r["backend"] == "local" and False, r["expected_tokens"]))
    return out


# --------------------------------------------------------------------------- fuel gauge: calibration, runway, cost
def calibrate_capacity(records: List[Dict[str, Any]], min_total_pct: float = 5.0) -> Dict[str, Any]:
    """Estimate tokens per 1% of the claude 5-hour window from measured, non-concurrent records.

    Vendors report percentages, not token counts, and claude's percentage is an integer, so a single
    task is too coarse: tokens and percent deltas are summed over all usable records and reported only
    when the summed percent movement reaches min_total_pct.
    """
    tok = 0
    pct = 0.0
    n = 0
    for r in records:
        if r.get("backend") != "claude" or r.get("concurrent"):
            continue
        d = ((r.get("window_delta") or {}).get("claude") or {}).get("session_pct_delta")
        if d is None or d <= 0:
            continue
        tok += int(r["tokens_est"])
        pct += float(d)
        n += 1
    if pct < min_total_pct:
        return {"status": "insufficient_data", "records": n, "total_pct": pct, "needed_pct": min_total_pct}
    per_pct = tok / pct
    return {"status": "ok", "records": n, "tokens_per_pct": round(per_pct), "window_capacity_tokens": round(per_pct * 100)}


def runway(snap_backend: Dict[str, Any], calib: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Estimated tokens left in the claude 5-hour window (None without a calibration)."""
    if calib.get("status") != "ok" or snap_backend.get("session_pct_used") is None:
        return None
    left_pct = 100.0 - snap_backend["session_pct_used"]
    return {"window": "5h", "remaining_pct": left_pct, "est_tokens_left": round(left_pct * calib["tokens_per_pct"])}


def headroom_report(snap: Dict[str, Any], floor: float = DEFAULT_FLOOR_PCT) -> List[Dict[str, Any]]:
    """Per-backend fuel gauge: tightest window remaining and whether it is under the floor."""
    out = []
    for b, data in snap.get("backends", {}).items():
        head = _remaining_pct(data, b)
        out.append({"backend": b, "status": data.get("status"), "tightest_remaining_pct": None if head is None else round(head, 1),
                    "constrained": head is not None and head < floor})
    return out










# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Measured agent usage telemetry (MISSION-035)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="Collect a zero-token usage snapshot")
    s.add_argument("--backends", default="claude,agy,codex")
    s.add_argument("--out", help="Write snapshot JSON to this file")
    d = sub.add_parser("diff", help="Window deltas between two snapshot files")
    d.add_argument("before")
    d.add_argument("after")
    h = sub.add_parser("headroom", help="Current headroom per backend; flags constrained windows")
    h.add_argument("--floor", type=float, default=DEFAULT_FLOOR_PCT)
    r = sub.add_parser("recommend", help="Rank backends for a task class (advisory)")
    r.add_argument("--task-class", required=True)
    r.add_argument("--floor", type=float, default=DEFAULT_FLOOR_PCT)
    c = sub.add_parser("calibrate", help="Estimate tokens-per-percent of the claude 5-hour window from measured records")
    c.add_argument("--min-pct", type=float, default=5.0)
    e = sub.add_parser("evaluate", help="Held-out error of the cost predictor vs the static ceiling")
    e.add_argument("--min-train", type=int, default=8)
    args = ap.parse_args()

    if args.cmd == "snapshot":
        snap = snapshot([b.strip() for b in args.backends.split(",") if b.strip()])
        text = json.dumps(snap, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
    elif args.cmd == "diff":
        print(json.dumps(diff(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text())), indent=2))
    elif args.cmd == "headroom":
        snap = snapshot(["claude", "agy", "codex"])
        for b, data in snap["backends"].items():
            head = _remaining_pct(data, b)
            flag = "CONSTRAINED" if head is not None and head < args.floor else "ok"
            print(f"{b:<7} status={data.get('status'):<12} tightest_window_remaining={'?' if head is None else f'{head:.1f}%':<7} {flag}")
    elif args.cmd == "recommend":
        snap = snapshot(["claude", "agy", "codex"])
        print(json.dumps(recommend(load_measured(), snap, args.task_class, args.floor), indent=2))
    elif args.cmd == "calibrate":
        calib = calibrate_capacity(load_measured(), args.min_pct)
        out = {"calibration": calib}
        if calib["status"] == "ok":
            out["runway"] = runway(snapshot(["claude"])["backends"]["claude"], calib)
        print(json.dumps(out, indent=2))
    elif args.cmd == "evaluate":
        print(json.dumps(evaluate_predictor(load_measured(), args.min_train), indent=2))


if __name__ == "__main__":
    sys.exit(main())
