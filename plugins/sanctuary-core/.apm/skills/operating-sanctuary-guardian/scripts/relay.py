#!/usr/bin/env python3
"""relay.py - Predecessor Relay for the Protocol 128 boot briefing (MISSION-025).

Purpose: turn the continuity data the repository already records (lease CHECKIN events, mission verdicts and evidence, mission priorities)
    into a short section an incoming agent reads FIRST: who did what last (with vendor), what is already established (cite it, do not
    re-run it), what is open, and any explicit handoff note a predecessor left via `manage_coordination.py checkin --handoff`.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 fast-boot context).
Key Input Dependencies (all optional; missing files just shorten the section):
    - .agent/learning/events.jsonl          CHECKIN events (tracked; the durable copy) and
    - .agent/coordination/leases.jsonl      CHECKIN events (untracked local log): agent_id, mission_id, outcome_status, pr_number, timestamp, handoff
    - LEARNING/research/missions/*.json     verdict, evidence, updated_at
    - LEARNING/research/index.jsonl         open missions by priority
    - LEARNING/research/inbox.jsonl         captured ideas (count only)
Key Outputs: a dict (build_relay) and a markdown section (render_markdown), about 3 KB, used by guardian_wakeup.py.
Usage:
    python3 relay.py show [--root DIR] [--handovers 5] [--results 10] [--json]
Key Functions: vendor_of(), build_relay(), render_markdown(), main().
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

VENDORS = {"claude-code": "Anthropic", "claude": "Anthropic", "antigravity": "Google", "agy": "Google", "gemini": "Google",
           "codex": "OpenAI", "mlx": "local Qwen", "qwen": "local Qwen"}
FULL_HANDOFFS = 3          # the newest handovers show a long handoff note (the successor needs the next step, which comes last)
MAX_HANDOFF_SHOWN = 400
MAX_HANDOFF_OLDER = 120


def vendor_of(agent_id: str) -> str:
    """Vendor of an agent id (unknown ids are reported as 'unknown')."""
    return VENDORS.get((agent_id or "").lower(), "unknown")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pointer(root: Path, mission: Dict[str, Any]) -> str:
    """Where to read the evidence: a path citation, else the mission's experiment log, else the mission manifest that holds the evidence."""
    evidence = (mission.get("evidence") or [None])[-1]
    cite = evidence.get("citation") if isinstance(evidence, dict) else None
    if isinstance(cite, str) and "/" in cite and cite.endswith((".md", ".json")):
        return cite
    num = re.sub(r"\D", "", str(mission.get("id", "")))
    exp = f"LEARNING/research/experiments/mission-{num}/README.md"
    if num and (root / exp).exists():
        return exp
    return f"LEARNING/research/missions/{mission.get('id')}.json"


def _checkins(root: Path) -> List[Dict[str, Any]]:
    """CHECKIN events from the tracked events log and the (untracked, possibly shared-root) leases log, de-duplicated."""
    sources = [root / ".agent/learning/events.jsonl"]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import manage_coordination
        sources.append(manage_coordination.shared_root(root) / ".agent/coordination/leases.jsonl")
    except Exception:  # noqa: BLE001 - the relay must still render without the coordination module
        sources.append(root / ".agent/coordination/leases.jsonl")
    seen, out = set(), []
    for src in sources:
        for e in _read_jsonl(src):
            if e.get("event_type") != "CHECKIN":
                continue
            key = (e.get("mission_id"), e.get("agent_id"), e.get("timestamp"))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out


def build_relay(root: Path, handovers: int = 5, results: int = 10) -> Dict[str, Any]:
    """Collect the relay data for a project root."""
    root = Path(root)
    checkins = _checkins(root)
    checkins.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    recent = [{"when": (e.get("timestamp") or "")[:16].replace("T", " "), "agent": e.get("agent_id"), "vendor": vendor_of(e.get("agent_id", "")),
               "mission": e.get("mission_id"), "outcome": e.get("outcome_status"), "pr": e.get("pr_number"),
               "handoff": _clip(e.get("handoff") or "", MAX_HANDOFF_SHOWN if i < FULL_HANDOFFS else MAX_HANDOFF_OLDER) or None}
              for i, e in enumerate(checkins[:handovers])]

    missions: List[Dict[str, Any]] = []
    mdir = root / "LEARNING/research/missions"
    if mdir.exists():
        for p in sorted(mdir.glob("MISSION-*.json")):
            try:
                missions.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    resolved = sorted([m for m in missions if m.get("status") == "RESOLVED" and m.get("verdict")], key=lambda m: m.get("updated_at", ""), reverse=True)
    established = [{"id": m["id"], "verdict": m["verdict"], "pointer": _pointer(root, m)} for m in resolved[:results]]

    index = _read_jsonl(root / "LEARNING/research/index.jsonl")
    open_ = sorted([x for x in index if x.get("status") != "RESOLVED"], key=lambda x: -float(x.get("priority_score") or 0))[:3]
    inbox = len(_read_jsonl(root / "LEARNING/research/inbox.jsonl"))
    return {"handovers": recent, "established": established, "dispatchable": _dispatchable(root), "fuel": _fuel(root),
            "open": [{"id": x["id"], "score": x.get("priority_score"), "status": x.get("status")} for x in open_], "inbox_items": inbox}


def _fuel(root: Path) -> Optional[Dict[str, Any]]:
    """Fuel gauge from the cached usage snapshot (no CLI calls at boot); None when absent or older than 6 hours."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import usage_hook
        snap = usage_hook.load_latest(Path(root))
        mod = usage_hook._probe()
        if not snap or mod is None:
            return None
        return {"as_of": snap.get("timestamp", "?")[:16].replace("T", " "), "gauges": mod.headroom_report(snap)}
    except Exception:
        return None


def _dispatchable(root: Path, count: int = 3) -> List[str]:
    """Open missions that can run in parallel now (disjoint from active leases and each other); empty if the flight board is unavailable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import manage_coordination
        return manage_coordination.suggest(Path(root), count)["selected"]
    except Exception:
        return []


def render_markdown(relay: Dict[str, Any]) -> str:  # relay: the dict from build_relay
    """Render the relay as a briefing section (kept to roughly 3 KB)."""
    lines = ["## III-B. Predecessor Relay (read before choosing work; cite results, do not re-run them)"]
    if relay["handovers"]:
        lines.append("* **Last handovers** (newest first):")
        for h in relay["handovers"]:
            pr = f" PR #{h['pr']}" if h.get("pr") else ""
            note = f" | handoff: {h['handoff']}" if h.get("handoff") else " | no handoff note"
            lines.append(f"  - {h['when']} `{h['agent']}` ({h['vendor']}) {h['mission']} {h['outcome']}{pr}{note}")
    if relay["established"]:
        lines.append("* **Established results (cite the evidence; do not repeat the work):**")
        for r in relay["established"]:
            lines.append(f"  - {r['id']} {r['verdict']} -> {r['pointer']}")
    if relay["open"]:
        top = "; ".join(f"{o['id']} ({o['score']}, {o['status']})" for o in relay["open"])
        lines.append(f"* **Open work by priority:** {top} | captured ideas in inbox: {relay['inbox_items']}")
    fuel = relay.get("fuel")
    if fuel:
        gauges = " · ".join(f"{g['backend']} {'?' if g['tightest_remaining_pct'] is None else str(int(g['tightest_remaining_pct'])) + '%'}"
                            + (" LOW" if g["constrained"] else "") for g in fuel["gauges"])
        low = [g["backend"] for g in fuel["gauges"] if g["constrained"]]
        hint = f" | LOW ({', '.join(low)}): prefer other backends" if low else ""
        lines.append(f"* **Fuel gauge** (tightest window left, {fuel['as_of']}; `usage_probe.py headroom`): {gauges}{hint}")
    else:
        lines.append("* **Fuel gauge:** no reading under 6 h old; run `usage_probe.py headroom` (evaluating-agent-work) before dispatching")
    if relay.get("dispatchable"):
        lines.append("* **Dispatchable in parallel now (disjoint from active leases and each other; `manage_coordination.py suggest`):** " + ", ".join(relay["dispatchable"]))
    lines.append("* **When you finish:** `manage_coordination.py checkin --mission <ID> --agent <you> --handoff \"what you did, decisions, what NOT to redo, next step (keep it under about 400 characters)\"`")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Predecessor relay for the boot briefing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="Print the relay section")
    show.add_argument("--root", default=".", help="Project root (default: current directory)")
    show.add_argument("--handovers", type=int, default=5)
    show.add_argument("--results", type=int, default=10)
    show.add_argument("--json", action="store_true", help="Print the raw data instead of markdown")
    args = ap.parse_args(argv)
    relay = build_relay(Path(args.root).resolve(), args.handovers, args.results)
    print(json.dumps(relay, indent=2) if args.json else render_markdown(relay))
    return 0


if __name__ == "__main__":
    sys.exit(main())
