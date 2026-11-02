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
      --evidence-refs LEARNING/topics/automaton_analysis/astra_automaton_opinion.md \
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


MODEL_COST_WEIGHTS = {
    # Frontier models (4.0x - 4.5x)
    "gpt-6-astra": 4.0,
    "astra": 4.0,
    "claude-opus": 4.0,
    "opus": 4.0,
    "claude-fable": 4.5,
    "fable": 4.5,
    "gemini-4": 4.0,
    "o1": 3.5,
    "o3": 4.0,

    # Standard models (1.0x - 1.5x)
    "gpt-6.1-sol": 1.5,
    "sol": 1.5,
    "gpt-6-luna": 1.0,
    "luna": 1.0,
    "claude-sonnet": 1.0,
    "sonnet": 1.0,
    "gemini-pro": 1.0,
    "pro": 1.0,
    "gpt-4o": 1.0,

    # Cheap models (0.1x - 0.2x)
    "claude-3-5-haiku": 0.2,
    "claude-haiku": 0.2,
    "haiku": 0.2,
    "gemini-3.8-flash": 0.1,
    "gemini-flash": 0.1,
    "flash": 0.1,
    "gpt-4o-mini": 0.15,
    "mini": 0.15,
    "local-llama": 0.05,
    "llama": 0.05,
    "qwen": 0.05
}


def get_model_cost_weight(model_name: str) -> float:
    clean = model_name.lower().strip()
    for k, w in MODEL_COST_WEIGHTS.items():
        if k in clean or clean in k:
            return w
    return 1.0


def update_ledger_projection(root: Path, assessment: dict) -> Path:
    ledger_path = root / ".agent" / "learning" / "cognitive_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_assessments": 0,
        "cumulative_score": 0,
        "average_fitness": 0.0,
        "dimension_totals": {
            "useful_outcome": 0,
            "learning_depth": 0,
            "verification_rigor": 0,
            "token_economy": 0,
            "anti_asch_integrity": 0
        },
        "agents": {},
        "recent_assessments": []
    }

    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    data["total_assessments"] = data.get("total_assessments", 0) + 1
    raw = assessment["raw_score"]
    data["cumulative_score"] = data.get("cumulative_score", 0) + raw
    data["average_fitness"] = round(data["cumulative_score"] / (data["total_assessments"] * 15.0), 3)

    # Dimension totals
    dim_totals = data.setdefault("dimension_totals", {})
    for dim, val in assessment["dimensions"].items():
        dim_totals[dim] = dim_totals.get(dim, 0) + val

    # Agent breakdown
    agent_id = assessment["agent_id"]
    agents = data.setdefault("agents", {})
    agent_stat = agents.setdefault(agent_id, {
        "assessments": 0,
        "cumulative_score": 0,
        "average_fitness": 0.0
    })
    agent_stat["assessments"] += 1
    agent_stat["cumulative_score"] += raw
    agent_stat["average_fitness"] = round(agent_stat["cumulative_score"] / (agent_stat["assessments"] * 15.0), 3)

    # Recent list (keep last 20)
    recent = data.setdefault("recent_assessments", [])
    recent.insert(0, {
        "assessment_id": assessment["assessment_id"],
        "timestamp": assessment["timestamp"],
        "task_id": assessment["task_id"],
        "agent_id": agent_id,
        "raw_score": raw,
        "fitness": assessment["cognitive_fitness"],
        "notes": assessment["notes"]
    })
    data["recent_assessments"] = recent[:20]
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

    raw_score = sum(dimensions.values())
    cognitive_fitness = round(raw_score / 15.0, 3)

    model_weight = get_model_cost_weight(args.model)
    cost_adjusted_roi = round(cognitive_fitness / model_weight, 3)

    bloat_warning = None
    if model_weight >= 3.0 and (args.token_economy <= 1 or args.useful_outcome <= 1):
        bloat_warning = f"CRITICAL: Expensive model '{args.model}' (weight {model_weight}x) produced bloated/low-value output without advancing core objectives."

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
        "max_score": 15,
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
