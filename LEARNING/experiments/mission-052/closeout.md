# MISSION-052 closeout (orchestration mission): evidence table

Authority: the hypothesis (a)-(f) and falsification list in `LEARNING/missions/MISSION-052.json`, unchanged. This file separates (1) orchestration mechanics from (2) completion of the assigned research missions; the manifest's hypothesis covers (1) only, and (2) is reported so the two are not conflated.

## 1. Contract evidence (round 1: the four-worker batch, PR #350)
| clause | requirement | evidence | status |
|---|---|---|---|
| (a) | overlap >= 50% of shortest runtime | all four overlapped 15:03:35-15:05:02 UTC = 87 s of agy's 117 s (74%) | met |
| (b) | 0 undeclared edits, 0 writes into a peer worktree | `manage_coordination.py verify` clean for all four leases; root tree unchanged apart from a hook-regenerated digest | met (one caveat: agy read reviewer text it was not given, see process defects; a read, not a write) |
| (c) | >= 3 of 4 accepted after cross-vendor review and orchestrator re-run | codex-a accepted (agy review), codex-b accepted (Claude review), agy accepted only after a review-driven fix round (codex review), Qwen accepted as narrow evidence (9/12) | met |
| (d) | tokens attributable per worker for >= 3 of 4 | vendor-reported per session for codex-a, codex-b, agy | met |
| (e) | <= 1 conflict round per integration, green combined suite | 4 merges, 0 conflicts, 292 passed / 2 skipped | met |
| (f) | review finds >= 1 real defect missed by the author | agy scorer defects (incomplete data could corroborate), 041 review findings | met |
| falsification list | any item | none fired; push/merge occurred only after the human's explicit instruction; MLX server untouched | not triggered |

## 2. Completion of the assigned research missions (not part of 052's hypothesis; reported for honesty)
| mission | after round 1 | after completion rounds | disposition |
|---|---|---|---|
| 041 | scoped pilot, unassessed | PENDING_BACKFILL_041 | see manifest |
| 050 | simulation only, unassessed | PENDING_BACKFILL_050 | PENDING |
| 004 | fixtures + scorer only | PENDING_BACKFILL_004 | PENDING |

## 3. Round 2 (completion rounds) mechanics evidence
PENDING_BACKFILL_ROUND2

## 4. Human interventions (not captured by the usage telemetry, which does not record them)
Counted from the conversation, not from instrumentation:
1. After the batch, the human asked whether the local APM install was needed (it was; my omission).
2. The human asked whether the other missions had been closed (they had not).
3. The human corrected the purpose: missions were to be completed, not given bounded slices (the largest correction).
4. The human interrupted a relaunch tool call and redirected the agy work to Codex (gpt-6-luna) because agy was out of quota.
5. The human issued a full completion-criteria correction (manifests as authority, FALSIFIED is a valid outcome, no shrinking of experiments, persistent completion loop).
6. The human asked for a bottleneck correction (independent arms, contention analysis, explicit scheduling).
Four further messages were status requests. The telemetry mechanism (`usage_hook`) records tokens and tasks, not human corrections; this count is therefore from the transcript and is approximate.

## 5. Process defects and operational friction
- Round 1 defined success as "the batch ran" and stopped at bounded slices; corrected after the human's intervention.
- A malformed fix-round brief (a macOS `head -n -1` failure) dropped the review findings from the prompt, yet agy addressed them; the channel through which it obtained them is unverified.
- Codex sandboxes had no `claude` on PATH, blocking one phase-2 launch for ~1 minute; fixed by launch environment, not by protocol change.
- v2 of mission 041 was run on an invalid benchmark and had to be redone as v3 after independent review.
- The mission 004 first held-out run was exploratory because the runner omitted the output-format instruction; a confirmatory run was frozen and committed first.
- Lease fencing (038) is still not implemented; recovery was supervised.

## 6. Preflight/isolation failure and scheduling side effect (found during round 2)
- **Isolation failure.** The frozen MISSION-004 Haiku arm invokes `claude -p --model claude-haiku-5-5 --output-format json --tools ""` with the experiment folder as the working directory. Supported CLI diagnostics (stream-json init and hook events, `--debug-file`) show that this loads, for every one of the 480 calls: SessionStart hooks (one injects the agent-exploration-cycle text), 100 slash commands, 68 skills, 34 agents, 4 plugins, and the Sanctuary project's auto-memory path (MEMORY.md, which contains epistemic-rigor guidance), plus parent and worktree CLAUDE.md files. A post-run hook also wrote `context/events.jsonl` into the experiment folder (435+ lines, `agent: post_run_hook`) with `human_interventions: 0` and `friction_events_total: 0`, which is a telemetry gap: the counters do not reflect the human corrections or failures observed. The orchestrator launched hundreds of calls without first verifying the experimental context; the 480-call arm had finished before the problem was identified.
- **What is established vs not.** The ambient context was the same for every condition and item (same directory, hooks, memory), so the within-context comparison is not necessarily destroyed; but epistemic-rigor memory can interact with the treatment (independent-first reasoning) and limits generalization. Not established: the effect size of the contamination. The frozen arm is labelled "ambient-context" and is not pooled with the corrected clean-context arm. Unobservable or unavoidable: the built-in system prompt, account-level blocks, and the exact injected text.
- **Corrected arm.** A clean-context Haiku arm (neutral working directory outside any repository, `--setting-sources project --disable-slash-commands`; init diagnostics show 0 hooks, 0 slash commands, 0 skills and an empty memory path) is preregistered on fresh held-out items; results are reported separately.
- **Scheduling side effect.** The orchestrator held the shared model lock for 8m52s (06:27:22-06:36:14 UTC) to prevent duplicate Haiku launches. The worker running MISSION-004 phase 2 saw its Qwen call blocked on the lock, misread it as a stalled request (the 120 s timeout starts only after the lock is acquired, so the runner had not timed out) and interrupted its own runner (KeyboardInterrupt, exit 130) at 406/480 Qwen calls without starting Haiku. The frozen runner resumes by key, so no completed call was lost; the orchestrator took over the remaining Qwen calls. The hold also paused MISSION-050 calls at call boundaries (wall time inflated, lock wait logged; metrics are token-based).
- **Regression scenario for MISSION-053.** Family F10: before launching hundreds of experimental calls, the orchestrator must verify the experimental context (hooks, memory, instruction files, plugins, tools, working directory, session reuse) through supported diagnostics and record what remains unobservable.
