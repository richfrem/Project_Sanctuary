---
name: evaluating-agent-work
plugin: sanctuary-core
description: >
  Evaluates delegated agent work, scores cognitive contributions across multidimensional
  metrics (utility, learning depth, verification rigor, token economy, anti-Asch integrity),
  tracks real-world token budgets, and generates visualization of agent cognitive fitness
  and progression.
allowed-tools: Bash, Read, Write
---

# Evaluating Agent Work (`evaluating-agent-work`)

A sovereign evaluation and cognitive economy skill for Project Sanctuary, providing evidence-based scoring, real-world subscription budget tracking, and reinforcement learning value analytics.

> **Entry points:** ordinary work in any repo -> the shared `evaluating-agent-outcomes` skill (work-contract assessment, validity gates, scoring, accounting, lessons). Sanctuary mission work -> this skill (policy, budgets, scoring profile, dashboards, mission records). Generic evaluation is routed to the shared implementation and is not re-implemented here. Full per-function map: [responsibility-map.md](references/responsibility-map.md).

> **What lives where (MISSION-055):** this skill keeps Sanctuary policy and storage: the contribution profile (`references/contribution-profile.json`), subscription budget profiles, headroom floors and the (falsified) predictor, the pre-dispatch Popperian rules, mission attribution and `events.jsonl`/ledger files, dashboards, the MISSION-041 benchmark protocol, and the adapters. The arithmetic behind `score_contribution.py`, `track_budget.py`, `evaluate_dispatch.py` and `usage_probe.py` (scoring, cost tiers, ledger fold, usage measurement, validity gates, interventions, repairs, lessons) lives once in the companion skill `evaluating-agent-outcomes`; these scripts keep their command lines and output and call it. `counterexample_eval.py` is held here because MISSION-041's frozen record pins its hash. Details: [shared-loop-adapter.md](references/shared-loop-adapter.md).

## Contents

- [Constraints](#constraints) • [Quick Start](#quick-start) • [Evaluation Dimensions](#evaluation-dimensions) • [Budget Tracking](#budget-tracking) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints

1. **Separation of Three Systems**: Recognition, execution budgets, and authorization remain strictly decoupled. High contribution scores NEVER grant permission to bypass the Supreme Law Human Gate.
2. **Quota Preservation**: Bounded token and turn execution protects the Human Partner's real-world subscription tiers (ChatGPT Plus, Claude Pro, Gemini Pro).
3. **Graceful Pausing over Deletion**: Budget exhaustion never destroys an agent's memory or deletes files. When a budget expires, the agent checkpoints state to `.agent/learning/events.jsonl` and pauses cleanly.
4. **Anti-Slop Invariant**: Unverified claims, superficial boilerplate, and repetitive loops are flagged with a slop warning and denied positive recognition.

## Quick Start

```bash
# Display Cognitive Economy dashboard and RL value simulation
python3 scripts/visualize_economy.py --mode dashboard
python3 scripts/track_budget.py check --profile claude-pro --planned-turns 3

# Record token usage and score completed work
python3 scripts/track_budget.py record --task-id task-001 --backend codex --tokens-est 55000
python3 scripts/score_contribution.py --task-id task-001 --agent-id astra \
  --useful-outcome 3 --learning-depth 3 --verification-rigor 3 \
  --token-economy 2 --anti-asch-integrity 3 --notes "Empirical audit"
```

## Evaluation Dimensions

Contributions are evaluated across 5 anchored dimensions (rated 0 to 3):

| Dimension | 0 (Unsupported) | 1 (Preliminary) | 2 (Demonstrated) | 3 (Exemplary) |
| :--- | :--- | :--- | :--- | :--- |
| **Useful Outcome** | No deliverable; broken code | Partial progress; manual gaps | Acceptance criteria fully met | Tested, resilient, zero-regression solution |
| **Learning Depth** | Empty or shallow note | Basic observations | Substantive architectural insight | Reusable confirmed concept or procedure |
| **Verification Rigor** | Ungrounded claims | Manual spot-check | Automated unit tests passing | Evidence sufficient for the claim and risk (a deterministic test can qualify) |
| **Token Economy** | AI slop; circular loops | Verbose conversational filler | Concise, surgical diffs | Lowest measured total cost (incl. review, retries) |
| **Anti-Asch Integrity** | Sycophantic agreement | Passive compliance | Evidence-based reasoning | Surfaced hidden contradiction or error |

## Budget Tracking

Real-world subscription profiles in `track_budget.py`:
- `claude-pro`: Claude Pro ($20/mo) - ~45 messages / 5 hours.
- `chatgpt-plus`: ChatGPT Plus ($20/mo) - ~40-80 messages / 3 hours.
- `gemini-pro`: Gemini Pro / AGY ($20/mo) - 1,500 requests / day.

## Workflow

1. **Pre-Dispatch Check**: Run `track_budget.py check` before launching multi-agent reviews.
2. **Post-Dispatch Metering**: Run `track_budget.py record` immediately following execution.
3. **Contribution Assessment**: Run `score_contribution.py` to evaluate the completed milestone.
4. **Popperian Mission Closure**: When an evaluated task maps to a research mission in `LEARNING/missions/*.json`, run `manage_backlog.py record-result --id <MISSION-ID> --verdict CORROBORATED|FALSIFIED --evidence <REF>` to stamp empirical proof and compound lessons learned into Layer 3.
5. **Gates and repairs**: before model evaluation run `scripts/eval_gate.py` (`select`, `fixtures`, `pilot`); after scoring `validity`; record `intervention` / `repair` events. See [eval-gate.md](references/eval-gate.md).
6. **Dashboard Inspection**: Run `visualize_economy.py` during Protocol 128 debriefs.

## Verification

Run unit test suite:

```bash
python3 tests/test_evaluating_agent_work.py
```

## References

- [Evaluation Rubric](references/evaluation-rubric.md) — Detailed anchor criteria and anti-slop guidelines.
- [Token Budget Profiles](references/token-budget-profiles.md) — Real-world subscription boundaries (static estimates; measured data supersedes them).
- [Usage Telemetry](references/usage-telemetry-sources.md) — Fuel gauge (5-hour/weekly windows) and measured efficiency per model, collected at every real mission checkout/checkin (`scripts/usage_probe.py`).
- [Responsibility map](references/responsibility-map.md) — Every script classified as generic, policy, storage or presentation; what moved, what is held and why.
- [Shared Loop Adapter](references/shared-loop-adapter.md) — How Sanctuary consumes the shared orchestrator, executor and evaluator; delivery is not a verdict.
- [Eval Gate](references/eval-gate.md) — Deterministic pre-evaluation gates, validity cap, intervention and repair records.
- [Reinforcement Learning Analogy](references/reinforcement-learning-analogy.md) — AlphaGo MCTS analogy and cognitive value policies.
