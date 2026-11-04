#!/usr/bin/env python3
"""usage_probe.py - Measured agent usage telemetry (MISSION-035).

Purpose:
    Deterministic, orchestrator-run before/after usage snapshots across the model backends
    (claude, agy, codex, local) so token and quota-window consumption is measured, not estimated
    or self-reported by the worker model. Feeds track_budget.py (source=measured) and a
    per-task-class cost predictor / backend recommender.
Layer: sanctuary-core / evaluating-agent-work.
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
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|secret)\"?\s*[:=]\s*\"?[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)hf_[A-Za-z0-9]{20,}"),
]
STATIC_CEILINGS = {"claude": 60000, "codex": 80000, "agy": 150000, "local": 80000}
DEFAULT_FLOOR_PCT = 15.0


def get_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def assert_no_secrets(obj: Any) -> None:
    """Raise ValueError if any credential-looking string appears in obj (serialised)."""
    blob = json.dumps(obj, default=str)
    for pat in SECRET_PATTERNS:
        if pat.search(blob):
            raise ValueError(f"credential-like pattern in usage record ({pat.pattern[:24]}...)")


# --------------------------------------------------------------------------- parsers
def parse_claude_usage(text: str) -> Dict[str, Any]:
    """Parse headless `claude -p /usage` text into session (5-hour) and weekly percent used."""
    out: Dict[str, Any] = {"session_pct_used": None, "session_resets": None,
                           "week_pct_used": None, "week_resets": None}
    m = re.search(r"Current session:\s*(\d+)%\s*used\s*[·\-]\s*resets\s*(.+)", text)
    if m:
        out["session_pct_used"], out["session_resets"] = int(m.group(1)), m.group(2).strip()
    m = re.search(r"Current week \(all models\):\s*(\d+)%\s*used\s*[·\-]\s*resets\s*(.+)", text)
    if m:
        out["week_pct_used"], out["week_resets"] = int(m.group(1)), m.group(2).strip()
    return out


def parse_agy_usage(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse `agy -p /usage --output-format json` into one row per quota bucket."""
    rows: List[Dict[str, Any]] = []
    groups = (((obj or {}).get("command") or {}).get("data") or {}).get("groups") or []
    for g in groups:
        for b in g.get("buckets") or []:
            frac = b.get("remaining_fraction")
            if frac is None:
                continue
            window = b.get("window") if b.get("window") in ("weekly", "5h") else "other"
            rows.append({"group": g.get("name", ""), "bucket": b.get("name", ""), "window": window,
                         "remaining_pct": round(float(frac) * 100.0, 3), "resets_at": b.get("reset_time", "")})
    return rows


def reported_tokens(backend: str, obj: Dict[str, Any]) -> int:
    """Tokens the CLI itself reports for a headless call (used to assert probes cost 0)."""
    u = (obj or {}).get("usage") or {}
    if backend == "agy":
        return int(u.get("total_tokens", 0) or 0)
    if backend == "claude":
        return int(u.get("input_tokens", 0) or 0) + int(u.get("output_tokens", 0) or 0)
    return 0


def read_codex_latest(sessions_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Newest rollout that has a token_count event; returns window percents and session totals.

    Scans newest-first because TUI-only sessions write rollouts with no token_count. Reads only
    rollout-*.jsonl under sessions_dir; never opens auth files.
    """
    sessions_dir = sessions_dir or Path.home() / ".codex" / "sessions"
    files = sorted(sessions_dir.glob("**/rollout-*.jsonl"), reverse=True)
    for f in files[:40]:
        last = None
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if '"token_count"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if (rec.get("payload") or {}).get("type") == "token_count":
                        last = rec
        except OSError:
            continue
        if last is None:
            continue
        p = last["payload"]
        rl = p.get("rate_limits") or {}
        tot = ((p.get("info") or {}).get("total_token_usage")) or {}
        return {
            "five_hour_pct_used": (rl.get("primary") or {}).get("used_percent"),
            "weekly_pct_used": (rl.get("secondary") or {}).get("used_percent"),
            "five_hour_resets_at": (rl.get("primary") or {}).get("resets_at"),
            "weekly_resets_at": (rl.get("secondary") or {}).get("resets_at"),
            "session_total_tokens": tot.get("total_tokens"),
            "source_timestamp": last.get("timestamp"),
            "source_file": f.name,
        }
    return None


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def claude_tokens_between(project_dir: Path, start: str, end: str,
                          own_session: Optional[str] = None) -> Dict[str, Any]:
    """Sum assistant-message token usage in [start, end] across Claude transcripts.

    concurrent=True when a transcript other than own_session wrote messages in the interval
    (window-percent deltas are then shared with that activity and must be treated as noisy).
    """
    t0, t1 = _parse_ts(start), _parse_ts(end)
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": 0}
    active = set()
    for f in project_dir.glob("*.jsonl"):
        sid = f.stem
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if '"usage"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    ts = _parse_ts(rec.get("timestamp", ""))
                    usage = (rec.get("message") or {}).get("usage")
                    if not ts or not usage or not (t0 <= ts <= t1):
                        continue
                    active.add(sid)
                    if own_session is None or sid == own_session:
                        for k in tot:
                            tot[k] += int(usage.get(k, 0) or 0)
        except OSError:
            continue
    # Without a known own session, two or more active transcripts means shared activity.
    others = active - {own_session} if own_session else (active if len(active) > 1 else set())
    return {**tot, "concurrent": bool(others), "other_active_sessions": len(others)}


# --------------------------------------------------------------------------- snapshot / diff
def _run(argv: List[str], timeout: int = 60) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True).stdout


def snapshot(backends: List[str], runner: Callable[..., str] = _run, now: Optional[str] = None,
             codex_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Collect a usage snapshot. Every collector is zero-token; failures degrade to 'unavailable'."""
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap: Dict[str, Any] = {"timestamp": now, "backends": {}}
    for b in backends:
        try:
            if b == "claude":
                data = parse_claude_usage(runner(["claude", "-p", "/usage"]))
                data["status"] = "ok" if data["session_pct_used"] is not None else "unparsed"
            elif b == "agy":
                raw = json.loads(runner(["agy", "-p", "/usage", "--output-format", "json"]))
                tokens = reported_tokens("agy", raw)
                rows = parse_agy_usage(raw)
                data = {"status": "ok" if rows else "unparsed", "buckets": rows, "probe_tokens": tokens}
                if tokens:
                    data["status"] = "probe_cost_nonzero"
            elif b == "codex":
                r = read_codex_latest(codex_dir)
                data = {"status": "ok", **r} if r else {"status": "unavailable"}
            else:
                data = {"status": "unsupported"}
        except Exception as exc:  # noqa: BLE001 - probes must never crash the orchestrator
            data = {"status": "unavailable", "error": type(exc).__name__}
        snap["backends"][b] = data
    assert_no_secrets(snap)
    return snap


def diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Window deltas between two snapshots, per backend."""
    out: Dict[str, Any] = {"from": before["timestamp"], "to": after["timestamp"]}
    bb, ab = before["backends"], after["backends"]
    if "claude" in bb and "claude" in ab:
        c0, c1 = bb["claude"], ab["claude"]

        def d(k):
            return None if c0.get(k) is None or c1.get(k) is None else c1[k] - c0[k]
        out["claude"] = {"session_pct_delta": d("session_pct_used"), "week_pct_delta": d("week_pct_used")}
    if "agy" in bb and "agy" in ab:
        idx = {(r["group"], r["window"]): r for r in bb["agy"].get("buckets", [])}
        rows = []
        for r in ab["agy"].get("buckets", []):
            prev = idx.get((r["group"], r["window"]))
            if prev:
                rows.append({"group": r["group"], "window": r["window"],
                             "remaining_pct_delta": round(r["remaining_pct"] - prev["remaining_pct"], 3)})
        out["agy"] = rows
    if "codex" in bb and "codex" in ab:
        x0, x1 = bb["codex"], ab["codex"]

        def dc(k):
            return None if x0.get(k) is None or x1.get(k) is None else round(x1[k] - x0[k], 3)
        out["codex"] = {"five_hour_pct_delta": dc("five_hour_pct_used"), "weekly_pct_delta": dc("weekly_pct_used"),
                        "stale": x0.get("source_timestamp") == x1.get("source_timestamp")}
    return out


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


def load_prices(path: Optional[Path] = None) -> Dict[str, Any]:
    """Price table (USD per million tokens) from references/model-prices.json. Unknown prices are null."""
    path = path or Path(__file__).resolve().parent.parent / "references" / "model-prices.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError):
        return {}


def cost_usd(model: str, tokens_in: int, tokens_out: int, prices: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Dollar cost when the model has a known price; None otherwise (never guessed)."""
    prices = load_prices() if prices is None else prices
    p = prices.get(model) or {}
    if p.get("input_per_mtok") is None or p.get("output_per_mtok") is None:
        return None
    return round(tokens_in / 1e6 * p["input_per_mtok"] + tokens_out / 1e6 * p["output_per_mtok"], 6)


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
