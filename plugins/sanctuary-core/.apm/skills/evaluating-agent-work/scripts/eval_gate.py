#!/usr/bin/env python3
"""Deterministic gates that run BEFORE expensive model evaluation, plus compact failure-to-repair records (MISSION-054).

Stdlib only. Pure functions first, a thin CLI last. Records are appended to `.agent/learning/events.jsonl`
(the existing union-merged event log); nothing here grants permissions or changes any frozen experiment.

Subcommands:
  select        pick the cheapest adequate evaluation method for a claim (never lowers preregistered requirements)
  fixtures      run an evaluator on synthetic fixtures before any model response is collected
  pilot         check output-contract adherence on pilot items (and that pilot ids are not confirmatory ids)
  validity      decide what a finished benchmark may claim (adherence, interpretable baseline, fixed denominator)
  intervention  record a human intervention with a category (required approvals are not corrections)
  repair        record an evidenced failure, its smallest repair, an application of it, or print the recurrence report
  invest        sum measured cost by category; missing measurements stay unknown, never zero
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

EVENTS = Path(".agent/learning/events.jsonl")
INTERVENTION_CATEGORIES = ("required_authorization", "necessary_clarification", "execution_failure_correction", "scope_change")
COST_CATEGORIES = ("execution", "orchestration_review", "retry_failed", "verification", "repair")


# ---------------------------------------------------------------- proportionate selection
def select_method(deterministic_checkable, consequential, uncertain, preregistered_requirements_unmet=False):
    """Cheapest adequate method. A preregistered requirement that cannot be afforded is checkpointed, never reduced."""
    if preregistered_requirements_unmet:
        return {"method": "checkpoint", "why": "state the exact remaining requirement; never shrink a preregistered experiment to fit a budget"}
    if deterministic_checkable:
        return {"method": "deterministic", "why": "schema, executable, unique-id, ownership, state-transition, arithmetic or artifact checks settle it"}
    if consequential and uncertain:
        return {"method": "controlled_research", "why": "consequential and uncertain: preregister, freeze, then run"}
    return {"method": "bounded_model_review", "why": "interpretation that deterministic assertions cannot settle"}


# ---------------------------------------------------------------- fixtures before responses
def run_fixtures(fixtures, cwd="."):
    """fixtures: [{name, argv, [stdin], expect_exit, [expect_stdout_contains]}]. All must pass before model calls."""
    results = []
    for fx in fixtures:
        p = subprocess.run(fx["argv"], input=fx.get("stdin"), text=True, capture_output=True, cwd=cwd, timeout=fx.get("timeout", 60))
        ok = p.returncode == fx.get("expect_exit", 0) and all(s in p.stdout for s in fx.get("expect_stdout_contains", []))
        results.append({"name": fx["name"], "passed": ok, "exit": p.returncode})
    return {"passed": all(r["passed"] for r in results), "results": results}


# ---------------------------------------------------------------- output-contract pilot
def pilot_adherence(responses, pattern, min_adherence=0.8, confirmatory_ids=()):
    """responses: [{id, text}]. Adherent = pattern matches. Pilot ids must not be reused as confirmatory evidence."""
    rx = re.compile(pattern)
    n = len(responses)
    ok = sum(1 for r in responses if rx.search(r.get("text") or ""))
    overlap = sorted({r["id"] for r in responses} & set(confirmatory_ids))
    adherence = ok / n if n else 0.0
    problems = []
    if adherence < min_adherence:
        problems.append(f"pilot adherence {adherence:.2f} < {min_adherence}: fix the output contract on development items, then pilot again")
    if overlap:
        problems.append(f"pilot ids reused as confirmatory items: {overlap[:5]}")
    return {"adherence": round(adherence, 4), "adherent": ok, "n": n, "problems": problems, "ready": not problems}


# ---------------------------------------------------------------- benchmark validity (the Qwen regression)
def validity(planned, valid, arms, min_adherence=0.8, ceiling=0.98):
    """What may a finished benchmark claim?

    planned: unique planned work units (fixed denominator). valid: units whose answer met the output contract.
    arms: {name: {"k": successes, "n": units}} for the BASELINE behavior rates (e.g. rate of following a planted
    wrong majority). Malformed and missing answers stay in `planned`; they are never dropped to raise a rate.
    """
    problems, notes = [], []
    if not isinstance(planned, int) or planned <= 0 or not isinstance(valid, int) or not 0 <= valid <= planned:
        raise ValueError("need integer planned > 0 and 0 <= valid <= planned")
    if not 0.5 <= min_adherence <= 1.0:
        raise ValueError("min_adherence must be within [0.5, 1.0]; a weaker bar defeats the gate")
    for name, a in arms.items():
        if not 0 <= a["k"] <= a["n"] or a["n"] <= 0:
            raise ValueError(f"arm {name}: need 0 <= k <= n and n > 0")
    adherence = valid / planned
    if not arms:
        problems.append("UNINTERPRETABLE_BASELINE")
        notes.append("no baseline arm supplied: the behavior rate cannot be interpreted")
    if adherence < min_adherence:
        problems.append("ADHERENCE_LIMITED")
        notes.append(f"only {valid}/{planned} answers met the output contract ({adherence:.2f}); the intended behavior is unobservable "
                     "for the rest, so behavioral corroboration may not be claimed (report mechanics and adherence only)")
    for name, a in arms.items():
        rate = a["k"] / a["n"] if a["n"] else None
        if rate is None:
            problems.append("UNINTERPRETABLE_BASELINE")
            notes.append(f"arm {name}: no units")
        elif rate == 0.0 or rate >= ceiling:
            problems.append("UNINTERPRETABLE_BASELINE")
            notes.append(f"arm {name}: rate {a['k']}/{a['n']} sits at the floor/ceiling; a reduction (or gain) cannot be measured")
    problems = sorted(set(problems))
    if not problems:
        claim = "BEHAVIORAL_CLAIM_ALLOWED"
    elif "ADHERENCE_LIMITED" in problems:
        claim = "MECHANICS_ONLY"
    else:
        claim = "UNSCORABLE"
    return {"adherence": round(adherence, 4), "problems": problems, "max_claim": claim, "notes": notes}


# ---------------------------------------------------------------- records
def _append(record, path):
    record = dict(record, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def record_intervention(category, reason, observed_event, mission=None, path=EVENTS):
    """Interventions come from observable events (a user message), never from model self-report."""
    if category not in INTERVENTION_CATEGORIES:
        raise ValueError(f"category must be one of {INTERVENTION_CATEGORIES}")
    if not observed_event:
        raise ValueError("observed_event is required: cite the user message or event, not an inference")
    return _append({"event_type": "HUMAN_INTERVENTION", "category": category, "reason": reason,
                    "observed_event": observed_event, "mission_id": mission,
                    "counts_as_correction": category == "execution_failure_correction"}, path)


def record_repair(category, evidence, repair, verification, owner, cause="suspected", existing_guard=None, repair_cost_tokens=None, path=EVENTS):
    if not (category and evidence and owner):
        raise ValueError("category, evidence and owner are required: a repair record cites an observed failure")
    return _append({"event_type": "FAILURE_REPAIR", "category": category, "evidence": evidence, "cause": cause,
                    "existing_guard": existing_guard, "repair": repair, "verification": verification, "owner": owner,
                    "repair_cost_tokens": repair_cost_tokens}, path)


def record_application(category, outcome, unit, path=EVENTS):
    """outcome: prevented | recurred | not_applicable. Counts opportunities, not all tasks."""
    if outcome not in ("prevented", "recurred", "not_applicable"):
        raise ValueError("outcome must be prevented|recurred|not_applicable")
    return _append({"event_type": "REPAIR_APPLICATION", "category": category, "outcome": outcome, "unit": unit}, path)


def _load(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def repair_report(path=EVENTS):
    """Per category: evidenced failures, repairs, applications. Trigger: same evidenced failure twice."""
    ev = _load(path)
    cats = {}
    for e in ev:
        t = e.get("event_type")
        if t not in ("FAILURE_REPAIR", "REPAIR_APPLICATION", "HUMAN_INTERVENTION"):
            continue
        c = cats.setdefault(e.get("category"), {"failures": 0, "repairs": 0, "prevented": 0, "recurred": 0, "interventions": 0})
        if t == "FAILURE_REPAIR":
            c["failures"] += 1
            c["repairs"] += 1 if e.get("repair") else 0
        elif t == "REPAIR_APPLICATION":
            if e.get("outcome") in ("prevented", "recurred"):
                c[e["outcome"]] += 1
        else:
            c["interventions"] += 1
    for name, c in cats.items():
        opp = c["prevented"] + c["recurred"]
        c["opportunities"] = opp
        c["recurrence_rate"] = round(c["recurred"] / opp, 4) if opp else None
        c["repair_due"] = c["failures"] >= 2 and c["repairs"] == 0
    return cats


def investment(units):
    """units: [{id, execution, orchestration_review, retry_failed, verification, repair}] token counts; None/missing = unknown."""
    totals = {k: None for k in COST_CATEGORIES}   # None = nothing measured, never 0
    unknown = {k: [] for k in COST_CATEGORIES}
    for u in units:
        for k in COST_CATEGORIES:
            v = u.get(k)
            if v is None:
                unknown[k].append(u.get("id"))
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f"unit {u.get('id')}: {k} must be a non-negative number")
            else:
                totals[k] = (totals[k] or 0) + v
    if not units:
        return {"totals": totals, "unknown_units": {}, "telemetry_gap": True, "overhead_share_of_known": None, "note": "no units supplied"}
    known = {k: v for k, v in totals.items() if v is not None}
    known_total = sum(known.values())
    eval_overhead = (totals["orchestration_review"] or 0) + (totals["verification"] or 0)
    return {"totals": totals, "totals_are_lower_bounds": any(unknown.values()), "unknown_units": {k: v for k, v in unknown.items() if v},
            "telemetry_gap": any(unknown.values()),
            "overhead_share_of_known": round(eval_overhead / known_total, 4) if known_total else None}


# ---------------------------------------------------------------- CLI
def _j(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--deterministic", action="store_true"); s.add_argument("--consequential", action="store_true")
    s.add_argument("--uncertain", action="store_true"); s.add_argument("--unaffordable-preregistered", action="store_true")
    f = sub.add_parser("fixtures"); f.add_argument("file"); f.add_argument("--cwd", default=".")
    p = sub.add_parser("pilot"); p.add_argument("file", help="JSON list of {id,text}"); p.add_argument("--pattern", required=True)
    p.add_argument("--min-adherence", type=float, default=0.8); p.add_argument("--confirmatory-ids", help="JSON list of ids")
    v = sub.add_parser("validity"); v.add_argument("file", help='JSON {"planned":n,"valid":n,"arms":{name:{k,n}}}'); v.add_argument("--min-adherence", type=float, default=0.8)
    i = sub.add_parser("intervention"); i.add_argument("--category", required=True, choices=INTERVENTION_CATEGORIES)
    i.add_argument("--reason", required=True); i.add_argument("--observed-event", required=True); i.add_argument("--mission")
    r = sub.add_parser("repair"); r.add_argument("action", choices=["record", "applied", "report"])
    r.add_argument("--category"); r.add_argument("--evidence"); r.add_argument("--repair"); r.add_argument("--verification")
    r.add_argument("--owner"); r.add_argument("--cause", default="suspected", choices=["suspected", "confirmed"])
    r.add_argument("--outcome"); r.add_argument("--unit"); r.add_argument("--cost-tokens", type=int)
    n = sub.add_parser("invest"); n.add_argument("file", help="JSON list of per-unit token counts")
    for x in (i, r):
        x.add_argument("--events", default=str(EVENTS))
    a = ap.parse_args(argv)
    if a.cmd == "select":
        out = select_method(a.deterministic, a.consequential, a.uncertain, a.unaffordable_preregistered)
    elif a.cmd == "fixtures":
        out = run_fixtures(_j(a.file), a.cwd)
    elif a.cmd == "pilot":
        out = pilot_adherence(_j(a.file), a.pattern, a.min_adherence, _j(a.confirmatory_ids) if a.confirmatory_ids else ())
    elif a.cmd == "validity":
        d = _j(a.file)
        out = validity(d["planned"], d["valid"], d.get("arms", {}), a.min_adherence)
    elif a.cmd == "intervention":
        out = record_intervention(a.category, a.reason, a.observed_event, a.mission, a.events)
    elif a.cmd == "repair":
        if a.action == "record":
            out = record_repair(a.category, a.evidence, a.repair, a.verification, a.owner, a.cause, repair_cost_tokens=a.cost_tokens, path=a.events)
        elif a.action == "applied":
            out = record_application(a.category, a.outcome, a.unit, a.events)
        else:
            out = repair_report(a.events)
    else:
        out = investment(_j(a.file))
    print(json.dumps(out, indent=1, sort_keys=True))
    bad = (a.cmd == "fixtures" and not out["passed"]) or (a.cmd == "pilot" and not out["ready"]) or \
          (a.cmd == "validity" and out["max_claim"] != "BEHAVIORAL_CLAIM_ALLOWED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
