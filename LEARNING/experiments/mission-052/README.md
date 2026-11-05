# MISSION-052: four-worker parallel batch with the existing stack

Status: **not started** (created 2026-11-05 so a clean session can run it). Start from [`brief.md`](./brief.md); it holds the human's instruction verbatim plus what changed since it was drafted.

## What this tests, and what it does not
It runs ONE bounded batch (Codex A, Codex B, agy, local Qwen2.5-Coder-7B) under a main-thread Claude orchestrator using **only existing tools**: flight board, `merge_prep.py`, `cli-agent`, `usage_hook.py delegated`, `validate_missions.py`. That is the "configuration alone" arm of MISSION-033's falsification criterion (an orchestrator skill is only warranted if configuration alone does not suffice). It is **not** 033's three-worker pilot: it makes no claim about 033's thresholds (60% wall-clock, 25% overhead, 80% review agreement), and any speedup claim needs a valid baseline.

## Preregistered predictions (written before any worker is launched; priors are the author's)
| # | Prediction | Prior |
|---|---|---|
| P1 | All four workers run concurrently: the overlap interval is at least 50% of the shortest worker's runtime (from session timestamps). | 85% |
| P2 | Zero undeclared edits outside leased paths and zero writes into a peer worktree (`manage_coordination.py verify` per worker, root tree unchanged). | 75% |
| P3 | At least 3 of 4 deliveries are accepted after cross-vendor review **and** the orchestrator's own re-run of the acceptance checks. | 60% |
| P4 | Measured tokens are attributable per worker session for at least 3 of 4 workers (vendor output, transcripts or rollouts; not self-report). The two Codex workers are the risk: the current recorder sums all rollouts in a window. | 70% |
| P5 | Accepted deliveries integrate onto a dedicated integration branch with at most 1 conflict round each after `merge_prep.py regenerate`, and the combined suite is green. | 70% |
| P6 | Review finds at least 1 real defect that the author's own self-assessment missed. | 90% |
| P7 | Qwen returns at least 1 narrow output that passes its deterministic checks with no orchestrator repair. | 60% |
| P8 | agy completes inside its cap without driving the Gemini weekly window (11% when this was written) to exhaustion. | 65% |

## Falsification (any one)
An undeclared edit, a clobbered file or a write into a peer worktree; fewer than 2 accepted deliveries; the workers were not actually concurrent; per-worker tokens attributable for fewer than 3 workers; any integration that needs more than 1 conflict round despite `merge_prep`, or leaves the combined suite red; any push, merge to `main` or publication without the human's explicit approval; the human-managed MLX server on `:8080` stopped or reconfigured; or review finds no defect in any of the four deliveries (then cross-vendor review added nothing at this size, recorded as such).

## Stopping conditions
A worker exceeds its token or time cap; a worker leaves its leased paths; a worker loops; a lease is lost; local memory pressure threatens the machine (swap was about 92% used when this was written); a backend window falls below 5% (reassign, do not push through).

## Measurements to record per worker
Start and end time; input, output and cache tokens and their source; turns, retries and interventions; the backend window before and after with its freshness; changed files against declared impact; the reviewer, the findings and the orchestrator's own re-run result; the five anchored scores; the empirical disposition (corroborated, falsified, unassessed, abandoned) kept separate from the quality scores.

## Results
(filled in by the executing session)
