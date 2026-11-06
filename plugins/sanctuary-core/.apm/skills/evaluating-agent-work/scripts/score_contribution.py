#!/usr/bin/env python3
"""
score_contribution.py - Multi-Dimensional Cognitive Value Scoring CLI
====================================================================

Purpose:
    Evaluates delegated agent contributions across 5 anchored dimensions:
      1. Useful Outcome (0-3)
      2. Learning Depth (0-3)
      3. Verification Rigor (0-3)
      4. Token Economy / Anti-Slop (0-3)
      5. Anti-Asch Epistemic Integrity (0-3)

    Appends an immutable event to `.agent/learning/events.jsonl` and
    updates the rebuildable projection in `.agent/learning/cognitive_ledger.json`.

Usage:
    python3 score_contribution.py --task-id task-001 --agent-id astra \
      --useful-outcome 3 --learning-depth 3 --verification-rigor 3 \
      --token-economy 2 --anti-asch-integrity 3 \
      --evidence-refs LEARNING/research/automaton_analysis/astra_automaton_opinion.md \
      --notes "Substantive empirical audit uncovering 59ms wakeup latency."
"""

import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


# Generic arithmetic (scoring, cost tiers, projection folding) is single-sourced in the companion skill
# `evaluating-agent-outcomes` (MISSION-055). This script keeps Sanctuary's profile (meanings of the five 0-3 dimensions),
# its storage (events.jsonl + cognitive_ledger.json), the event type and the CLI/text output.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared  # noqa: E402,F401
import scoring as _scoring  # noqa: E402
import accounting as _accounting  # noqa: E402

PROFILE = _scoring.load_profile(Path(__file__).resolve().parent.parent / "references" / "contribution-profile.json")
MODEL_COST_WEIGHTS = {k: v["weight"] for k, v in _accounting.MODEL_COST_TIERS.items()}   # compat view of the shared table


def get_model_cost_weight(model_name: str) -> float:
    return _accounting.get_model_cost_weight(model_name)


def empty_projection() -> dict:
    return _scoring.empty_projection(PROFILE)


def fold_assessment(data: dict, assessment: dict) -> dict:
    """Apply one CONTRIBUTION_ASSESSED event to the projection (shared by the live scorer and the rebuild)."""
    return _scoring.fold_assessment(PROFILE, data, assessment)


def rebuild_projection(assessments: list) -> dict:
    """Deterministically rebuild the projection from events (updated_at = last event time, not now), order-independent."""
    return _scoring.rebuild_projection(PROFILE, assessments)


def update_ledger_projection(root: Path, assessment: dict) -> Path:
    ledger_path = root / ".agent" / "learning" / "cognitive_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    data = empty_projection()
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    fold_assessment(data, assessment)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return ledger_path


def main():
    parser = argparse.ArgumentParser(description="Multi-Dimensional Cognitive Contribution Scorer")
    parser.add_argument("--task-id", required=True, help="Task identifier")
    parser.add_argument("--agent-id", default="autonomous-agent", help="Agent identifier (astra, antigravity, claude)")
    parser.add_argument("--model", default="standard", help="Model name used (e.g. gpt-6-astra, haiku, sonnet)")
    parser.add_argument("--useful-outcome", type=int, choices=[0, 1, 2, 3], required=True, help="Useful outcome rating (0-3)")
    parser.add_argument("--learning-depth", type=int, choices=[0, 1, 2, 3], required=True, help="Learning depth rating (0-3)")
    parser.add_argument("--verification-rigor", type=int, choices=[0, 1, 2, 3], required=True, help="Verification rigor rating (0-3)")
    parser.add_argument("--token-economy", type=int, choices=[0, 1, 2, 3], required=True, help="Token economy rating (0-3)")
    parser.add_argument("--anti-asch-integrity", type=int, choices=[0, 1, 2, 3], required=True, help="Anti-Asch integrity rating (0-3)")
    parser.add_argument("--notes", default="", help="Assessment notes and justification")
    parser.add_argument("--evidence-refs", default="", help="Comma-separated evidence file paths")
    parser.add_argument("--assessor", default="peer-reviewer", help="Assessor identity")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    root = get_project_root()
    assessment_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    dimensions = {
        "useful_outcome": args.useful_outcome,
        "learning_depth": args.learning_depth,
        "verification_rigor": args.verification_rigor,
        "token_economy": args.token_economy,
        "anti_asch_integrity": args.anti_asch_integrity
    }

    scored = _scoring.score(PROFILE, dimensions)
    raw_score = scored["raw_score"]
    cognitive_fitness = scored["fitness"]

    model_weight = get_model_cost_weight(args.model)
    cost_adjusted_roi = _scoring.cost_adjusted(cognitive_fitness, model_weight)
    bloat_warning = _scoring.bloat_warning(PROFILE, dimensions, args.model, model_weight)

    evidence_list = [e.strip() for e in args.evidence_refs.split(",") if e.strip()]

    assessment = {
        "schema_version": 1,
        "assessment_id": assessment_id,
        "timestamp": timestamp,
        "event_type": "CONTRIBUTION_ASSESSED",
        "task_id": args.task_id,
        "agent_id": args.agent_id,
        "model": args.model,
        "model_cost_weight": model_weight,
        "assessor": args.assessor,
        "dimensions": dimensions,
        "raw_score": raw_score,
        "max_score": scored["max_score"],
        "cognitive_fitness": cognitive_fitness,
        "cost_adjusted_roi": cost_adjusted_roi,
        "bloat_warning": bloat_warning,
        "evidence_refs": evidence_list,
        "notes": args.notes
    }

    # Append to events.jsonl
    events_file = root / ".agent" / "learning" / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(assessment) + "\n")

    # Update ledger projection
    ledger_file = update_ledger_projection(root, assessment)

    if args.format == "json":
        print(json.dumps(assessment, indent=2))
    else:
        print(f"Scored Contribution for [{args.agent_id}] ({args.model}) on Task [{args.task_id}]:")
        print(f"  Raw Score:          {raw_score} / 15")
        print(f"  Cognitive Fitness:   {cognitive_fitness:.1%} (RL Normalized)")
        print(f"  Model Cost Weight:   {model_weight}x")
        print(f"  Cost-Adjusted ROI:   {cost_adjusted_roi:.2f}")
        if bloat_warning:
            print(f"  [!] {bloat_warning}")
        print(f"  Dimensions:")
        print(f"    - Useful Outcome:        {args.useful_outcome} / 3")
        print(f"    - Learning Depth:        {args.learning_depth} / 3")
        print(f"    - Verification Rigor:    {args.verification_rigor} / 3")
        print(f"    - Token Economy:         {args.token_economy} / 3")
        print(f"    - Anti-Asch Integrity:   {args.anti_asch_integrity} / 3")
        print(f"  Ledger Updated:      {ledger_file.relative_to(root)}")


if __name__ == "__main__":
    main()
