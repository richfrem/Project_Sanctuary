#!/usr/bin/env python3
"""merge_prep.py - regenerate shared generated files after a merge instead of hand-merging them (MISSION-034).

Regenerates from their sources:
  * LEARNING/index.jsonl and backlog.md  <- mission manifests (manage_backlog compile)
  * .agent/learning/cognitive_ledger.json          <- .agent/learning/events.jsonl (CONTRIBUTION_ASSESSED events)
  * chronicle/ledger.json                          <- chronicle/entries/*.md
The append-only log events.jsonl merges with git's `union` driver (see .gitattributes). The flight-board files are untracked runtime state.

The ledger is a projection. If the stored ledger claims MORE than its events imply (an orphaned score that was
edited into the aggregate but never logged as an event), this tool REPORTS it and leaves the file unchanged.
It never silently adjusts an aggregate. Exit codes: 0 ok, 3 drift reported, 2 usage error.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

SKILLS = Path(__file__).resolve().parents[2]
for _p in (SKILLS / "evaluating-agent-work" / "scripts", SKILLS / "managing-sovereign-memory" / "scripts",
           SKILLS / "managing-chronicles" / "scripts"):
    sys.path.insert(0, str(_p))
import score_contribution  # noqa: E402
import manage_backlog  # noqa: E402
import chronicle_manager  # noqa: E402

LEDGER_REL = ".agent/learning/cognitive_ledger.json"
EVENTS_REL = ".agent/learning/events.jsonl"
UNION_LOGS = [".agent/learning/events.jsonl"]
CHRONICLE_LEDGER_REL = "chronicle/ledger.json"
GENERATED = ["LEARNING/index.jsonl", "LEARNING/backlog.md", LEDGER_REL, CHRONICLE_LEDGER_REL]


def find_root(start: Path = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return p


def load_assessments(root: Path) -> List[Dict[str, Any]]:
    path = root / EVENTS_REL
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event_type") == "CONTRIBUTION_ASSESSED":
                out.append(e)
    return out


def implied_ledger(root: Path) -> Dict[str, Any]:
    return score_contribution.rebuild_projection(load_assessments(root))


def read_stored_sides(path: Path) -> List[Dict[str, Any]]:
    """Return the stored ledger(s): one dict normally, two (ours, theirs) when the file has conflict markers."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if "<<<<<<<" not in text:
        try:
            return [json.loads(text)]
        except json.JSONDecodeError:
            return []
    ours, theirs, mode = [], [], "common"
    for line in text.splitlines():
        if line.startswith("<<<<<<<"):
            mode = "ours"
        elif line.startswith("=======") and mode == "ours":
            mode = "theirs"
        elif line.startswith(">>>>>>>") and mode == "theirs":
            mode = "common"
        else:
            if mode in ("common", "ours"):
                ours.append(line)
            if mode in ("common", "theirs"):
                theirs.append(line)
    sides = []
    for side in (ours, theirs):
        try:
            sides.append(json.loads("\n".join(side)))
        except json.JSONDecodeError:
            pass
    return sides


def _comparable(d: Dict[str, Any]) -> Dict[str, Any]:
    c = {k: v for k, v in d.items() if k != "updated_at"}
    return c


def ledger_status(stored: Dict[str, Any], implied: Dict[str, Any]) -> str:
    """ok: equals what events imply; stale: claims less; drift: claims more (or an agent claims more)."""
    if _comparable(stored) == _comparable(implied):
        return "ok"
    if stored.get("total_assessments", 0) > implied["total_assessments"] \
            or stored.get("cumulative_score", 0) > implied["cumulative_score"]:
        return "drift"
    for agent, st in (stored.get("agents") or {}).items():
        imp = implied["agents"].get(agent, {"assessments": 0, "cumulative_score": 0})
        if st.get("assessments", 0) > imp["assessments"] or st.get("cumulative_score", 0) > imp["cumulative_score"]:
            return "drift"
    return "stale"


def missing_union_rules(root: Path) -> List[str]:
    ga = root / ".gitattributes"
    text = ga.read_text(encoding="utf-8") if ga.exists() else ""
    have = {m.group(1) for m in re.finditer(r"^(\S+)\s+merge=union\b", text, re.M)}
    return [p for p in UNION_LOGS if p not in have]


def _hash_dir(d: Path) -> Dict[str, str]:
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.glob("*.json"))} if d.exists() else {}


def regenerate(root: Path, apply: bool = True) -> Dict[str, Any]:
    root = Path(root)
    report: Dict[str, Any] = {"drift": False, "generated": [], "manifests_changed": []}

    ledger_path = root / LEDGER_REL
    implied = implied_ledger(root)
    sides = read_stored_sides(ledger_path)
    statuses = [ledger_status(s, implied) for s in sides] or ["missing"]
    if "drift" in statuses:
        status = "drift"
    elif all(s == "ok" for s in statuses) and len(sides) == 1:
        status = "ok"
    elif statuses == ["missing"]:
        status = "missing"
    else:
        status = "stale"
    action = "left-unchanged"
    if status == "drift":
        report["drift"] = True
        stored = max(sides, key=lambda s: s.get("total_assessments", 0))
        report["ledger_detail"] = {
            "stored_total_assessments": stored.get("total_assessments"), "stored_cumulative_score": stored.get("cumulative_score"),
            "implied_total_assessments": implied["total_assessments"], "implied_cumulative_score": implied["cumulative_score"],
            "note": "stored aggregate claims more than events.jsonl implies; resolve by logging the missing event or reverting the edit",
        }
    elif status in ("stale", "missing"):
        if apply:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(implied, indent=2), encoding="utf-8")
            action = "rebuilt"
            report["generated"].append(LEDGER_REL)
    elif status == "ok" and apply:
        action = "unchanged-ok"
    report["ledger"] = {"status": status, "action": action}

    before = _hash_dir(root / "LEARNING" / "missions")
    if apply and (root / "LEARNING" / "missions").exists():
        manage_backlog.BacklogManager(root).compile()
        report["generated"] += ["LEARNING/index.jsonl", "LEARNING/backlog.md"]
    after = _hash_dir(root / "LEARNING" / "missions")
    report["manifests_changed"] = [Path(k).name for k in after if before.get(k) != after[k]]

    entries_dir = root / "chronicle" / "entries"
    if apply and entries_dir.exists():
        chronicle_manager.write_ledger(entries_dir)
        report["generated"].append(CHRONICLE_LEDGER_REL)
    report["union_rules_missing"] = missing_union_rules(root)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("regenerate", "Regenerate shared generated files (report drift)"), ("check", "Report only; write nothing")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--root", default=None)
        sp.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args(argv)
    root = find_root(Path(args.root) if args.root else None)
    report = regenerate(root, apply=(args.cmd == "regenerate"))
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"ledger: {report['ledger']['status']} ({report['ledger']['action']})")
        if report["drift"]:
            d = report["ledger_detail"]
            print(f"DRIFT: stored {d['stored_total_assessments']} assessments / {d['stored_cumulative_score']} points, "
                  f"events imply {d['implied_total_assessments']} / {d['implied_cumulative_score']}. Not modified.")
        if report["manifests_changed"]:
            print("manifests changed by compile:", ", ".join(report["manifests_changed"]))
        if report["union_rules_missing"]:
            print("missing union rules in .gitattributes:", ", ".join(report["union_rules_missing"]))
        if report["generated"]:
            print("regenerated:", ", ".join(report["generated"]))
    return 3 if report["drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
