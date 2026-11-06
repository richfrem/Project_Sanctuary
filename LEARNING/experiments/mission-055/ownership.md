# MISSION-055 ownership and scope map

Worktrees: Sanctuary `.worktrees/claude-code-mission-055` (branch `feat/claude-code-mission-055`); companion `agent-plugins-skills/.worktrees/mission-055` (branch `feat/mission-055-learning-loop`, base `fb7bead2`), owner claude-code. No other open leases or companion worktrees at start.

| capability | source owner | caller | persistent state | change |
|---|---|---|---|---|
| Outcome loop, lesson retrieval, redispatch | companion `orchestrating-agent-work` (`work_loop.py`) | any repo; Sanctuary via guidance | `<root>/.agent-work/<id>/` trace, plan, findings | new |
| Launch checks (claim, cwd, executable, timeout, clean context, result JSON) | companion `cli-agent` (`run_agent.py`, `dispatch_guard.py`) | `work_loop.dispatch`; any caller passing flags | claim files, result JSON | wired into run_agent; dispatch_guard moved from orchestrator (shim left) |
| Findings, validity, cost, interventions, repairs, lessons | companion `evaluating-agent-outcomes` (`eval_core.py`) | orchestrator, Sanctuary adapter | explicit JSONL store inside a declared root | new; extracted from Sanctuary `eval_gate.py` |
| Sanctuary compat + policy | `sanctuary-core/evaluating-agent-work` (`eval_gate.py` shim, `sanctuary_adapter.py`) | guardian, agents | `.agent/learning/events.jsonl` | shim + adapter |
| Mission lifecycle, hypotheses, verdicts | `managing-sovereign-memory` / manifests | humans, `manage_backlog.py` | `LEARNING/missions/*.json` | unchanged |
| Scoring dashboards and budgets | `evaluating-agent-work` (`score_contribution.py`, `track_budget.py`, `usage_probe.py`) | guardian | `.agent/learning/*` | unchanged; usage_probe stays the headroom source |

## Scope boundaries
- Belongs to 055: integration, relocation of shipped guards (053 PR #748, 054 PR #360), generic contract, executor wiring, adapter, docs, installs.
- Stays with 053: baseline-vs-revised behavioral claim over ten scenario families (UNASSESSED).
- Stays with 054: Claim A (12 held-out scenarios, UNASSESSED) and Claim B (12 real work units, longitudinal, UNASSESSED).
- 036 (evidence contracts), 037 (invalidation), 038 (lease fencing): PROPOSED, not implemented. Claim files give duplicate-launch prevention only, not crash-safe fencing.
- 035: usage_probe telemetry exists and is reused; its per-class predictor is falsified and not used.
