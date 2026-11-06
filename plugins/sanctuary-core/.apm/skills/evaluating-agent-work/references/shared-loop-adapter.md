# Sanctuary adapter for the shared learning loop (MISSION-055)

Sanctuary does not own the generic loop. It consumes three skills from the companion repo (`agent-orchestration` plugin, installed by APM) and adds policy:

| Responsibility | Owner | Sanctuary adds |
|---|---|---|
| Outcome loop: contract, lesson retrieval, delegation, redispatch, integration | `orchestrating-agent-work` (`work_loop.py`, `references/learning-loop.md`) | Flight-board lease and worktree before dispatch; Human Gate: `authorization.may_merge` defaults to false |
| Launch checks: claim, cwd boundary, executable, timeout, clean context, structured result | `cli-agent` (`run_agent.py` flags, `dispatch_guard.py`) | Backend choice from `usage_probe.py headroom` (never dispatch an exhausted backend; do not probe it with model calls) |
| Findings, validity gates, cost, interventions, repairs, lessons | `evaluating-agent-outcomes` (`eval_core.py`) | `scripts/eval_gate.py` (compat adapter), `scripts/sanctuary_adapter.py`, store `.agent/learning/events.jsonl` |

Scoring, accounting and usage measurement are also single-sourced upstream now; see [responsibility-map.md](responsibility-map.md) for the per-function split, the one held exception and the compatibility evidence.

Kept local: mission manifests and lifecycle, hypotheses and falsification criteria, empirical verdicts (`manage_backlog.py record-result`), Protocol 128, chronicle, soul traces, research prioritization, benchmark definitions, and the scoring dashboards (`score_contribution.py`, `track_budget.py`, `visualize_economy.py`).

## Delivery is not a verdict
`sanctuary_adapter.py contract MISSION-NNN --delivery delivery.json` translates the manifest into a generic contract. The acceptance checks are engineering checks written in the delivery spec; falsification criteria are empirical and are not translated. `deliver` records `delivery_complete` and copies `empirical_assessment` from the manifest (UNASSESSED until `record-result` runs). The adapter has no code path that writes a verdict.

## Lessons versus procedures
Lessons (`lessons.py`, same events store) are short records with trigger, evidence, action, limits and recorded uses. Use `LEARNING/procedures/` and `record_procedure_outcome.py` only for protected, versioned playbooks with a recorded need. Do not create a mission per repair.

## Where this runs
The shared skills must be installed (`apm update --global`, then verify with the installed-skills check in guardian wakeup). `eval_gate.py` fails loudly if `evaluating-agent-outcomes` cannot be found. Controls enforced by code only apply to launches that pass the executor flags; direct CLI use bypasses them.
