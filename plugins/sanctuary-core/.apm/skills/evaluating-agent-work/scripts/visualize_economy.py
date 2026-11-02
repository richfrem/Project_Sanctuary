#!/usr/bin/env python3
"""
visualize_economy.py - Cognitive Economy & RL Fitness Visualization CLI
======================================================================

Purpose:
    Visualizes cognitive fitness, multi-dimensional score distributions,
    token economy, and reinforcement learning Q-value policies for autonomous
    AI agents collaborating in Project Sanctuary.

Usage:
    python3 visualize_economy.py
    python3 visualize_economy.py --mode rl-simulation
    python3 visualize_economy.py --mode mermaid
"""

import os
import sys
import json
import argparse
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def render_bar(val: float, max_val: float = 1.0, length: int = 20) -> str:
    if max_val <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, val / max_val))
    filled = int(round(pct * length))
    empty = length - filled
    return "█" * filled + "░" * empty


def render_dashboard(root: Path):
    ledger_path = root / ".agent" / "learning" / "cognitive_ledger.json"
    budget_path = root / ".agent" / "learning" / "budget_ledger.json"

    cog_data = {}
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                cog_data = json.load(f)
        except Exception:
            pass

    bud_data = {}
    if budget_path.exists():
        try:
            with open(budget_path, "r", encoding="utf-8") as f:
                bud_data = json.load(f)
        except Exception:
            pass

    total_assessments = cog_data.get("total_assessments", 0)
    avg_fitness = cog_data.get("average_fitness", 0.0)
    tokens_consumed = bud_data.get("total_tokens_consumed", 0)
    dim_totals = cog_data.get("dimension_totals", {})

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║            PROJECT SANCTUARY: COGNITIVE ECONOMY DASHBOARD          ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    fitness_bar = render_bar(avg_fitness, 1.0, 24)
    print(f"Overall Cognitive Fitness:  [{fitness_bar}] {avg_fitness:.1%}")
    print(f"Total Contributions Rated: {total_assessments}")
    print(f"Cumulative Tokens Tracked: {tokens_consumed:,}\n")

    print("─── Multi-Dimensional Competency Breakdown ─────────────────────────")
    dims = [
        ("useful_outcome", "Useful Outcome        "),
        ("learning_depth", "Learning / Depth      "),
        ("verification_rigor", "Verification Rigor    "),
        ("token_economy", "Token Economy / Restraint"),
        ("anti_asch_integrity", "Anti-Asch Truth Grounding")
    ]

    for key, label in dims:
        if total_assessments > 0:
            avg_dim = dim_totals.get(key, 0) / (total_assessments * 3.0)
            dim_bar = render_bar(avg_dim, 1.0, 16)
            pts = dim_totals.get(key, 0) / total_assessments
            print(f"  {label} : [{dim_bar}] {pts:.1f}/3 ({avg_dim:.0%})")
        else:
            print(f"  {label} : [░░░░░░░░░░░░░░░░] 0.0/3 (0%)")

    agents = cog_data.get("agents", {})
    if agents:
        print("\n─── Agent Cognitive Rosters ────────────────────────────────────────")
        for aid, astat in agents.items():
            afit = astat.get("average_fitness", 0.0)
            abar = render_bar(afit, 1.0, 12)
            print(f"  • {aid.upper():12} : [{abar}] {afit:.1%} ({astat.get('assessments', 0)} evaluations)")

    print()


def render_rl_simulation():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║      REINFORCEMENT LEARNING: POLICY EVALUATION & VALUE ESTIMATE    ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    print("Under the AlphaGo / MCTS formulation, agent actions are evaluated by their")
    print("expected Cognitive Return (Q-value) discounted by Token Expenditure:\n")

    strategies = [
        {
            "name": "Strategy A: Surgical Minimal-Diff + Automated Tests",
            "q_val": 0.92,
            "cost": "Low (1-2 turns, ~15k tokens)",
            "reward": "High Information Gain + Zero Regression Risk",
            "policy": "RECOMMENDED (Dominant Strategy)"
        },
        {
            "name": "Strategy B: Multi-Agent Bounded Peer Review (Astra / Codex)",
            "q_val": 0.88,
            "cost": "Medium (~50k tokens, 1 pass)",
            "reward": "Deep Architectural Consensus + Anti-Asch Calibration",
            "policy": "RECOMMENDED for High-Complexity Tasks"
        },
        {
            "name": "Strategy C: Fragmented Micro-Iterations & Verbose Churn",
            "q_val": 0.35,
            "cost": "High (10+ turns, ~120k tokens)",
            "reward": "Shallow Incremental Progress",
            "policy": "SUB-OPTIMAL (Token Inefficient)"
        },
        {
            "name": "Strategy D: Repetitive Tool Thrashing / Ungrounded Speculation",
            "q_val": -0.75,
            "cost": "Extremely High (Token Drain)",
            "reward": "AI Slop / Zero Verified Code",
            "policy": "INTERCEPTED by Execution Circuit Breaker"
        }
    ]

    for s in strategies:
        bar = render_bar(max(0.0, s["q_val"]), 1.0, 16)
        print(f"[{s['policy']}]")
        print(f"  {s['name']}")
        print(f"  Expected Value Q(s, a): [{bar}] {s['q_val']:+.2f}")
        print(f"  Resource Cost:          {s['cost']}")
        print(f"  Cognitive Yield:        {s['reward']}\n")


def render_mermaid():
    chart = """```mermaid
quadrantChart
    title Cognitive Value vs Resource Cost
    x-axis Low Token Cost --> High Token Cost
    y-axis Low Cognitive Value --> High Cognitive Value
    quadrant-1 High-Cost Breakthroughs
    quadrant-2 Optimal Sovereign Zone
    quadrant-3 Sub-optimal / Stagnant
    quadrant-4 Toxic Token Churn / Slop
    "Surgical Clean PR": [0.2, 0.9]
    "Automated Test Verification": [0.25, 0.85]
    "Astra Architectural Review": [0.45, 0.95]
    "Shallow Verbose Explanations": [0.75, 0.3]
    "Repetitive Tool Thrashing": [0.9, 0.1]
```"""
    print(chart)


def main():
    parser = argparse.ArgumentParser(description="Cognitive Economy Visualization CLI")
    parser.add_argument("--mode", choices=["dashboard", "rl-simulation", "mermaid"], default="dashboard")
    args = parser.parse_args()

    root = get_project_root()
    if args.mode == "dashboard":
        render_dashboard(root)
    elif args.mode == "rl-simulation":
        render_rl_simulation()
    elif args.mode == "mermaid":
        render_mermaid()


if __name__ == "__main__":
    main()
