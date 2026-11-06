#!/usr/bin/env python3
"""Thin Sanctuary policy adapter over the shared learning loop (MISSION-055).

Translates a mission manifest into the generic work contract and records the shared evaluator's findings as ENGINEERING DELIVERY,
kept separate from the EMPIRICAL assessment. Engineering acceptance never produces a verdict: verdicts come only from
`manage_backlog.py record-result` after the manifest's own falsification criteria are evaluated. This module has no code path that writes one.

  contract MISSION-NNN --delivery delivery.json [--may-merge]   print the generic contract (acceptance supplied by the delivery spec)
  deliver  MISSION-NNN FINDINGS.json                            append a delivery record (empirical state copied from the manifest)
  state    MISSION-NNN                                          delivery vs empirical state side by side
Local policy kept here: Human Gate (authorization.may_merge defaults to false), flight-board lease before dispatch, usage headroom as the
budget source, events.jsonl as the store.
"""
import argparse, json, sys
from pathlib import Path

import eval_gate as _g   # locates the shared skill and exposes its core

EVENTS = Path(".agent/learning/events.jsonl")


def _manifest(mission_id, root="."):
    f = Path(root) / "LEARNING" / "missions" / f"{mission_id}.json"
    return json.loads(f.read_text())


def empirical_state(manifest):
    """Copied from the manifest only. UNASSESSED unless a verdict was recorded by record-result."""
    return manifest.get("verdict") or "UNASSESSED"


def mission_to_contract(manifest, delivery, may_merge=False, max_attempts=3, max_tokens=None):
    """delivery: {"acceptance": [...], "failure_conditions": [...]} supplied by the caller: a mission's falsification criteria are
    empirical claims, not engineering acceptance checks, and are NOT translated."""
    budget = {"max_attempts": max_attempts}
    if max_tokens:
        budget["max_tokens"] = max_tokens
    return {"id": manifest["id"], "type": "sanctuary-mission", "goal": manifest["title"],
            "scope": {"paths": list(manifest.get("projected_impact", {}).get("paths", [])) or ["."]},
            "acceptance": delivery["acceptance"], "failure_conditions": delivery["failure_conditions"],
            "budget": budget, "authorization": {"may_merge": bool(may_merge)}}


def record_delivery(manifest, findings, path=EVENTS):
    rec = {"mission_id": manifest["id"], "delivery_complete": bool(findings.get("complete")), "delivery_recommended": findings.get("recommended"),
           "incomplete": findings.get("incomplete", []), "empirical_assessment": empirical_state(manifest),
           "note": "engineering delivery only; no verdict is derived from it"}
    return _g._c.append("delivery", rec, root=str(Path(path).parent), store=str(path), compat_event_type=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contract"); c.add_argument("mission"); c.add_argument("--delivery", required=True); c.add_argument("--may-merge", action="store_true")
    d = sub.add_parser("deliver"); d.add_argument("mission"); d.add_argument("findings")
    s = sub.add_parser("state"); s.add_argument("mission")
    a = ap.parse_args(argv)
    m = _manifest(a.mission, a.root)
    if a.cmd == "contract":
        out = mission_to_contract(m, json.loads(Path(a.delivery).read_text()), a.may_merge)
    elif a.cmd == "deliver":
        out = record_delivery(m, json.loads(Path(a.findings).read_text()), Path(a.root) / EVENTS)
    else:
        last = [e for e in _g._c.load(a.root, str(EVENTS)) if e.get("kind") == "delivery" and e.get("mission_id") == m["id"]]
        out = {"mission": m["id"], "manifest_status": m["status"], "empirical": empirical_state(m),
               "last_delivery": {k: last[-1][k] for k in ("delivery_complete", "delivery_recommended", "incomplete")} if last else None}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
