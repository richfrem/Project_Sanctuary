# MISSION-052: four-worker parallel batch with the existing stack

Status: **round 1 (four-worker batch) merged in PR #350; completion rounds in progress; see [`closeout.md`](./closeout.md) for the evidence table and verdict (pending until the assigned missions finish).** Start from [`brief.md`](./brief.md); it holds the human's instruction verbatim plus what changed since it was drafted.

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

## Results (executed 2026-11-05 by claude-code as orchestrator; base `852d11547`)

### Dispatch actually run
| worker | mission | worktree / branch | window (UTC) | outcome |
|---|---|---|---|---|
| codex-a | MISSION-041 | `.worktrees/codex-a-mission-041` | 15:03:05-15:13:56 | accepted; scoped result **misses** (+8.33pp vs +20pp) |
| codex-b | MISSION-050 | `.worktrees/codex-b-mission-050` | 15:03:05-15:14:22 | accepted as simulation evidence; thresholds **missed** |
| agy | MISSION-004 | `.worktrees/agy-mission-004` | 15:03:05-15:05:02 (+ fix round 15:17:43-15:21:29) | changes requested, fixed once, accepted |
| qwen-local | MISSION-052 subtask | `.worktrees/qwen-local-mission-052` | 15:03:35-15:08:11 | 9/12 deterministic pass |

Mission 023 was skipped (needs a second local model while swap was ~92%); 001 needs a human gate and a cloud VM.

### Empirical dispositions (kept separate from quality scores)
- MISSION-041: **unassessed at preregistered scale; scoped run missed a threshold** (6 components / 24 held-out mutants instead of 20 components; detection 100% vs 91.67% baseline = +8.33pp, needs +20pp; 0/6 false alarms; max 0.87 s). Evidence: `LEARNING/experiments/mission-041/results.md`. Manifest verdict NOT recorded: the run was a reduced scope.
- MISSION-050: **simulation candidate falsified; real agent untested.** Adaptive policy -20.6% (dev) / -26.2% (held-out) verified subgoals vs best fixed, 50-55% acknowledged progress lost (needs <=5%); overhead 4.6-4.7% passes, control loss 0% passes. Evidence: `LEARNING/experiments/mission-050/results.md`. Orchestrator finding the worker's own assessment did not isolate: `AdaptivePolicy.should_checkpoint` returns False whenever the observed interruption rate is 0 (cold start), so it never checkpoints before the first interruption (`adaptive_checkpoint.py:87-88`). Manifest verdict NOT recorded.
- MISSION-004: **unassessed** (fixtures + deterministic scorer only; no model was run).
- Qwen subtask: 9/12 exact-match numeric extractions with no repair; misses: MISSION-007 and 036 returned empty, MISSION-042 invented numbers (`results.json`, `driver.py`; the driver and the "ground truth" regex are the orchestrator's).

### Reviews and scores (0-5: outcome / learning / rigor / economy / anti-Asch)
| delivery | reviewer | verdict | scores | substantive defects found |
|---|---|---|---|---|
| codex-a 041 | agy (cross-vendor) | accept | 4/5/5/5/5 (agy's scale; generous, I spot-checked only that all 21 tests pass) | no subprocess process-group kill on timeout (med); 3 of 6 components use the example generator as the property generator, which caps the measurable gain (low); 6 not 20 components (med, disclosed) |
| codex-b 050 | claude-code (orchestrator, same vendor as the orchestrator, so NOT an independent re-score) | accept | 3/4/4/3/5 | cold-start no-checkpoint behaviour above |
| agy 004 | codex (cross-vendor) | changes-requested (2/2/2/3/2), then fixed by agy and accepted | high: incomplete data could report `hypothesis_corroborated: true`; high: acceptance metric did not measure an actual update; med: exact-40% float miss, normalization bias, `-Infinity` JSON | I re-ran the two-record case: now an explicit error, not corroboration. Codex did not re-review the fix. |
| qwen | deterministic check + orchestrator | accept as narrow-task evidence | n/a | 3/12 misses preserved |

### Measured usage (vendor-reported, per session; not self-report)
- codex-a: input 493,181 (460,928 cached), output 20,705. codex-b: input 1,073,175 (1,032,704 cached), output 26,448. Separated by their own `thread_id` in `codex exec --json` output, so the Codex pair IS attributable per worker (not via the window-sum recorder).
- agy: build 56,532 total; review of codex-a 98,100; fix round 180,072. Codex review of agy: input 290,440 (246,912 cached), output 4,335.
- Qwen: local, no vendor tokens recorded; 12 runs, 18-47 s each.
- Gemini weekly window 11.4% -> 9.9% across all agy runs; Codex window reading 58% at creation vs 87% now (stale/unexplained, not charged). MLX server on :8080 untouched and still responding.
- The stated 300k token caps were not enforced mid-run; counted without cache reads, both Codex workers were under them.

### Preregistered predictions
P1 met: all four overlapped 15:03:35-15:05:02 = 87 s, 74% of the shortest runtime (agy, 117 s). P2 met: `verify` clean for all four leases, root tree unchanged apart from the hook-regenerated boot digest. P3 met: 3 of 4 accepted after cross-vendor review (agy's only after a fix round) plus my re-run of each acceptance test. P4 met: tokens attributable for codex-a, codex-b and agy (3 of 4). P5 met: four merges, 0 conflicts, combined `evaluating-agent-work` + guardian suites 292 passed, 2 skipped. P6 met: real defects found in 004 (high) and 041 (med). P7 met: Qwen 9/12. P8 met: agy finished above 5%. No falsification condition fired. No speedup is claimed: there is no sequential baseline.

### Process defects to carry forward
- The agy fix-round brief was malformed (a macOS `head -n -1` failure dropped the review findings from the prompt, 621 bytes). agy nevertheless fixed the specific findings, so it obtained the reviewer's text through a channel I did not provide and did not inspect (possibly reading the shared scratchpad). Isolating reviewer output from authors needs a check.
- The 038 lease fencing is not implemented; recovery was supervised and no lease expired.
- Adding a mission-manifest `IN_PROGRESS` marker in each worker worktree was left uncommitted, as the brief required.

### Next
Review the integration branch, then decide on publication. Highest-value next experiment: run MISSION-041 at its preregistered 20-component scale, and give MISSION-050's policy a prior hazard estimate before the first interruption.
