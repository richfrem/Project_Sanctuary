# MISSION-022 experiment log

Hypothesis (MISSION-022): Qwen2.5-Coder-7B (4-bit, MLX, offline) completes >= 85% of bounded repository
discovery tasks in a plan -> tool -> observe -> final loop. Falsified if < 75% complete, or a call repeats > 4 times.

## Run history (honest record)

| Run | Harness | Tasks | Result | Validity |
|---|---|---|---|---|
| 1 | `agent_loop.py` v0 (JSON protocol, 2000-char reads, no guards) | 5 hand-written (`tasks.json`) | 1/5 correct (`results_run1_baseline.json`) | Informative failure: model repeated identical tool calls on 3 of 5 tasks, never finalised. `read_file` truncation also hid T3's answer. |
| 2 | `agent_loop.py` v1 (+12k reads, repeat suppression, forced final turn), tuned on run-1 failures | same 5 | 4/5 (`results.json`) | **Not valid evidence**: tuned on its own test set, N=5, forced-final scaffold, tasks hard-code paths, substring grader, `tasks.json` readable by the agent. |
| 3 | `local_agent_loop.py` (new loop, fixed tools) | same 5 as a **dev set only** | 4/5 (`dev_results_run3_new_loop.json`); T5 model error (answered `LEARNING/` only) | Dev check. Harness frozen after this run. |
| 4 | frozen `local_agent_loop.py` v1, seed 1337, temp 0 | 30 generated held-out tasks | **25/30 = 83.3%** (`heldout_results_seed1337_temp0.0.json`) | The pre-registered test. Landed in the 0.75-0.85 band = **inconclusive**, no verdict forced. |
| 5 | `local_agent_loop.py` v2 (search_text defect fixed), seed 2024, temp 0 | 30 generated tasks (12 prompts overlap seed 1337) | **22/30 = 73.3%** (`heldout_results_seed2024_temp0.0.json`) | Below the 75% line: **FALSIFIED** by the pre-registered rule. |

Review: agy (`gemini-4-argon-medium`) critiqued runs 1-2; adopted: held-out generated tasks with computed answers,
exact-set grading, answer key unreadable by the agent, intervention metrics, single threshold, reachable repetition limit.
Rejected: agy's `/`-path theory for T5 (trace showed the model passed a glob to `list_dir`; tool now returns an actionable error).

Harness fixes found during development (before any held-out run): tool-call syntax -- the MLX server strips the
`<tool_call>` special token, so the model emits `<{"name":...}`; `extract_calls` now scans for bare JSON calls.

## Pre-registration (written before the held-out run)

- Tasks: `tests/agent_eval_tasks.py`, seed 1337, 30 tasks (10 scalar / 10 locate / 10 dirset), answers computed from repo state.
- Settings: temperature 0.0 (primary), max 5 turns, repetition limit 4, ctx budget 8192 tokens. Harness frozen: no edits after the first held-out run.
- Prediction: correct_rate ~ 0.65 (scalar high, locate medium, dirset low: the model tends to give incomplete lists).
- Decision rule: correct_rate >= 0.85 and zero repetition falsifications -> CORROBORATED.
  correct_rate < 0.75 or any repetition falsification -> FALSIFIED. 0.75-0.85 -> reported as inconclusive to the human (no verdict forced).
- Always reported alongside: unassisted_correct_rate, repeat interventions, final-turn nudges, per-tier rates, tokens.

## Run 4 outcome (recorded as-is, before any change)

correct 0.833 (scalar 1.0 / locate 0.6 / dirset 0.9); completion 0.90; unassisted-correct 0.767; correct within 3 turns 0.633;
repeat interventions 1; tasks reaching the forced final turn 6; no repetition falsification; max prompt 3050 tokens; mean 22 s/task.
Prediction was 0.65, so the model beat the prediction. Failures: 3 locate tasks (L4, L6, L9) where the model passed a glob as
`search_text`'s `path` and the tool answered a misleading `(no matches)` -- a harness defect, not a model finding; L3 (read files
one by one until the turn limit); D9 (model misread a listing -- genuine model error).
The pytest gate (>= 0.85) correctly FAILED on this run.

## Run 5 pre-registration (written before running)

Run 4 is NOT re-run after the fix (that would be tuning on held-out). Instead: fix `search_text` (glob/missing path -> explicit
error, unit-tested), then evaluate on a fresh seed (2024) with the same thresholds and decision rule as above.
Prediction: ~0.83 (fix helps locate; model errors remain). Combined evidence (run 4 + run 5, 60 tasks) is reported; the verdict
uses the pre-registered rule on run 5 and must not contradict run 4 (run 4 stays inconclusive on its own).

## Run 5 outcome and verdict

correct 0.733 (scalar 1.0 / locate 0.4 / dirset 0.8); completion 0.867; unassisted-correct 0.70; correct within 3 turns 0.633;
repeat interventions 7; tasks reaching the forced final turn 7; no repetition falsification; max prompt 3012 tokens; 24 s/task.
Prediction was ~0.83: the model underperformed it. The `search_text` fix did not help: the model now received an ERROR but
followed the hint ("use find_files") back to the same glob call and then read mission files one at a time until the turn limit.
The hint itself misled the model (harness weakness found, not fixed: frozen). 2 of 8 failures are silent dirset errors (D9, D10).

**Verdict: FALSIFIED** (pre-registered: run 5 < 0.75). Combined 47/60 = 78.3%, never >= 0.85. The hypothesis fails as stated;
what holds: scalar single-hop 100% and most directory listings; what fails: multi-hop locate (40-60%).
Deviation from the originating task script: it presupposed CORROBORATED; the data did not support that and the verdict was not forced.

## Escalation-trigger analysis (60 tasks, runs 4+5)

Trigger = no final answer OR >=4 turns OR repeat intervention OR malformed reply OR forced final turn.
Flags 19/60 tasks; catches 10/13 failures (77%); 9 of 19 flagged tasks were correct (wasted escalations);
the 3 misses are silent directory-listing errors. Spawned MISSION-028 (parent: MISSION-007).

## Caveats

N=30 per run is small; seeds share 12 prompts; one model, one quantisation, temp 0 only; graders are strict (exact sets) which
penalises near-misses; first-turn behaviour (wasteful glob) may be promptable. Not tested: Haiku/Flash on the same tasks.
