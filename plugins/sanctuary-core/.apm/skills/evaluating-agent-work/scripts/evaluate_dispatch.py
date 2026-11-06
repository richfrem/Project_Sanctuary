#!/usr/bin/env python3
"""
evaluate_dispatch.py - Pre-Delegation Falsifiable Investment Evaluator
=====================================================================

Purpose:
    Applies the Popperian scientific method and investment ROI analysis
    BEFORE delegating tasks to external CLI agents or subagents.
    Ensures that every agent dispatch:
      1. Has a clear, testable, and falsifiable hypothesis.
      2. Establishes objective falsification criteria.
      3. Selects the most cost-effective model tier (cheap vs frontier).
      4. Frontier escalation is approved only after >= 2 failed attempts (--failures).

Key Input Dependencies:
    Appends PRE_DISPATCH_EVALUATION events to .agent/learning/events.jsonl
    under the project root (nearest ancestor containing .git).

Usage:
    python3 evaluate_dispatch.py --task "..." --hypothesis "..." \
      --falsification-criteria "..." [--model gpt-6-astra] [--planned-turns 2] \
      [--failures 0] [--format text|json]

Function Index:
    get_project_root, get_model_tier, evaluate_investment, main
"""

import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone


# Cost tiers are single-sourced in the companion skill `evaluating-agent-outcomes` (MISSION-055). The Popperian pre-dispatch
# investment rules below (falsifiable hypothesis, frontier only after repeated failures) are Sanctuary policy and stay here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared  # noqa: E402,F401
import accounting as _acc  # noqa: E402

MODEL_COST_TIERS = _acc.MODEL_COST_TIERS


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_model_tier(model_name: str) -> dict:
    return _acc.get_model_tier(model_name)


ESCALATION_FAILURE_THRESHOLD = 2


def evaluate_investment(hypothesis: str, falsification: str, model_info: dict, planned_turns: int, failures: int = 0) -> dict:
    """Evaluates whether the dispatch represents a smart, falsifiable investment.

    Args:
        hypothesis: Falsifiable prediction for the dispatch.
        falsification: Observable condition that would disprove the hypothesis.
        model_info: Tier record from get_model_tier (tier, weight, desc).
        planned_turns: Expected turn budget.
        failures: Failed attempts so far; frontier is permitted only when >= 2.

    Returns:
        Dict with status, is_falsifiable, model_tier, cost_weight, recommendations.
    """
    recommendations = []
    status = "APPROVED"

    # Falsifiability check
    hypo_words = len(hypothesis.split())
    fals_words = len(falsification.split())

    is_falsifiable = True
    if fals_words < 5:
        is_falsifiable = False
        recommendations.append("Falsification criteria too vague. Provide concrete, observable test assertions.")
        status = "NEEDS_REFINEMENT"

    tier = model_info["tier"]
    weight = model_info["weight"]

    # Model tier economy check
    if tier == "frontier":
        if planned_turns > 3:
            recommendations.append(f"Frontier model requested for > 3 turns ({planned_turns}). High token burn risk on subscription quota.")
            status = "WARNING"
        if hypo_words < 8:
            recommendations.append("Task description lacks architectural depth required to justify frontier model cost. Consider downgrading to a standard or cheap model.")
            status = "DOWNGRADE_RECOMMENDED"
        # Escalation gate: frontier needs repeated failures. A falsifiability failure keeps its status.
        if failures < ESCALATION_FAILURE_THRESHOLD:
            recommendations.append("Frontier model requested with failures < 2. Quota conservation rule requires routing to cheap tier (Gemini Flash / Claude Haiku / local MLX) before frontier escalation.")
            if is_falsifiable:
                status = "DOWNGRADE_REQUIRED"
        else:
            recommendations.append(f"Frontier escalation approved after {failures} failed attempts.")
            if is_falsifiable:
                status = "APPROVED"
    elif tier == "cheap":
        recommendations.append("High token economy. Ideal for broad reconnaissance, syntax fixes, or targeted queries.")

    return {
        "status": status,
        "is_falsifiable": is_falsifiable,
        "model_tier": tier,
        "cost_weight": weight,
        "recommendations": recommendations
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-Delegation Falsifiable Investment Evaluator")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--hypothesis", required=True, help="Falsifiable hypothesis or expected outcome")
    parser.add_argument("--falsification-criteria", required=True, help="Exact observable condition that would prove hypothesis false")
    parser.add_argument("--model", default="claude-3-5-sonnet", help="Target model identifier")
    parser.add_argument("--planned-turns", type=int, default=1, help="Expected turn budget")
    parser.add_argument("--failures", type=int, default=0, help="Failed attempts so far (frontier allowed at >= 2)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    root = get_project_root()
    model_info = get_model_tier(args.model)
    eval_result = evaluate_investment(
        args.hypothesis, args.falsification_criteria, model_info, args.planned_turns, failures=args.failures
    )

    dispatch_id = str(uuid.uuid4())
    record = {
        "dispatch_id": dispatch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "hypothesis": args.hypothesis,
        "falsification_criteria": args.falsification_criteria,
        "model": args.model,
        "model_tier": model_info["tier"],
        "cost_weight": model_info["weight"],
        "planned_turns": args.planned_turns,
        "investment_decision": eval_result["status"],
        "recommendations": eval_result["recommendations"]
    }

    # Append to events.jsonl
    events_file = root / ".agent" / "learning" / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event_type": "PRE_DISPATCH_EVALUATION",
            **record
        }) + "\n")

    if args.format == "json":
        print(json.dumps(record, indent=2))
    else:
        print(f"Pre-Delegation Investment Evaluation [{eval_result['status']}]:")
        print(f"  Task:               {args.task}")
        print(f"  Model Requested:    {args.model} ({model_info['desc']})")
        print(f"  Cost Weight:        {model_info['weight']}x ({model_info['tier'].upper()} tier)")
        print(f"  Hypothesis:         {args.hypothesis}")
        print(f"  Falsification Gate: {args.falsification_criteria}")
        if eval_result["recommendations"]:
            print("  Guidance / Recommendations:")
            for r in eval_result["recommendations"]:
                print(f"    - {r}")


if __name__ == "__main__":
    main()
