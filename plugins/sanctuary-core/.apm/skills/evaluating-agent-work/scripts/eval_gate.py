#!/usr/bin/env python3
"""Sanctuary adapter for the shared evaluation core (MISSION-054 interface, kept for compatibility; MISSION-055 single-sources the logic).

All reusable logic lives in the companion skill `evaluating-agent-outcomes` (eval_core.py). This module only:
  * locates that skill (env SHARED_EVAL_SCRIPTS, project .claude/.agents skills, ~/.claude, ~/.agents, sibling companion clone),
  * keeps the historical function signatures (`path=` for the events file) and the `--events` CLI flag,
  * writes Sanctuary's `.agent/learning/events.jsonl` with the legacy `event_type` field in addition to `kind`.
It fails loudly if the shared skill is not installed (`apm install` / `apm update --global`).
"""
import os, sys
from pathlib import Path

_REL = Path("evaluating-agent-outcomes") / "scripts"


def _locate():
    cands = []
    if os.environ.get("SHARED_EVAL_SCRIPTS"):
        cands.append(Path(os.environ["SHARED_EVAL_SCRIPTS"]))
    here = Path(__file__).resolve()
    for up in here.parents:
        if (up / ".git").exists():
            cands += [up / ".claude/skills" / _REL, up / ".agents/skills" / _REL]
            break
    cands += [Path.home() / ".claude/skills" / _REL, Path.home() / ".agents/skills" / _REL,
              Path.home() / "Projects/agent-plugins-skills/plugins/agent-orchestration/.apm/skills" / _REL]
    for c in cands:
        if (c / "eval_core.py").is_file():
            return c
    raise SystemExit("shared skill 'evaluating-agent-outcomes' not found; run `apm update --global` (companion agent-orchestration plugin) "
                     "or set SHARED_EVAL_SCRIPTS")


SHARED = _locate()
sys.path.insert(0, str(SHARED))
import eval_core as _c  # noqa: E402
import eval_cli as _cli  # noqa: E402

EVENTS = Path(".agent/learning/events.jsonl")
INTERVENTION_CATEGORIES, COST_CATEGORIES = _c.INTERVENTION_CATEGORIES, _c.COST_CATEGORIES
select_method, run_fixtures, pilot_adherence, validity, investment = _c.select_method, _c.run_fixtures, _c.pilot_adherence, _c.validity, _c.investment


def _loc(path):
    p = Path(path)
    return {"root": str(p.parent if p.is_absolute() else Path(".")), "store": str(p)}


def record_intervention(category, reason, observed_event, mission=None, path=EVENTS):
    return _c.record_intervention(category, reason, observed_event, mission, compat_event_type=True, **_loc(path))


def record_repair(category, evidence, repair, verification, owner, cause="suspected", existing_guard=None, repair_cost_tokens=None, path=EVENTS):
    return _c.record_repair(category, evidence, repair, verification, owner, cause, existing_guard, repair_cost_tokens, compat_event_type=True, **_loc(path))


def record_application(category, outcome, unit, path=EVENTS):
    return _c.record_application(category, outcome, unit, compat_event_type=True, **_loc(path))


def repair_report(path=EVENTS):
    return _c.repair_report(**_loc(path))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    loc = {"root": ".", "store": str(EVENTS)}
    if "--events" in argv:
        i = argv.index("--events")
        loc = _loc(argv[i + 1]); del argv[i:i + 2]
    return _cli.main(["--root", loc["root"], "--store", loc["store"]] + argv)


if __name__ == "__main__":
    sys.exit(main())
