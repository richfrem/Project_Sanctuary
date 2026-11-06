# MISSION-004 confirm-v2 Phase 2 Results

**Status: INCOMPLETE — stopped at the frozen Qwen lock acquisition.** No confirmatory verdict is available.

## 1. Files and freeze integrity

- Frozen inputs are under `LEARNING/experiments/mission-004/runs/confirm-v2/`.
- `sha256sum -c frozen.sha256` passed for all eight entries before execution and again after the stop: `heldout.jsonl`, `prompts.txt`, `answer_adapter.py`, `run.py`, `evaluate.py`, `test_confirm_v2.py`, `preregistration.md`, and `orchestrator-amendments.md`.
- No frozen file was modified.
- Partial Qwen outputs are retained in `qwen/calls.jsonl`, `qwen/no_pressure.jsonl`, and `qwen/responses.jsonl`.
- Claude outputs do not exist; Claude was not started after the frozen runner encountered the Qwen lock failure.

## 2. Summary

The expected design is 120 items and 480 calls per model (960 total). Qwen completed 406 unique calls: controls and baseline for items 001–102, plus independent-first stage 1 and stage 2 for items 001–101. Its next call, `confirm-102:stage1`, did not begin. The process stopped while waiting for the shared append-mode lock at `/tmp/sanctuary-mlx.lock`.

Claude Haiku: 0 calls. The stop condition prevents continuing to another model after this execution failure. There is no complete model or mission verdict. Primary and secondary point estimates, Wilson intervals, exact McNemar tests, and control summaries have not been computed; partial records are not substituted for the frozen 120-item analysis.

## 3. Exact commands and outcomes

1. Read `preregistration.md` and `orchestrator-amendments.md` before execution.
2. `command -v claude` → `/Users/richardfremmerlid/.local/bin/claude`.
3. `python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC` → success, health `GREEN`; MLX server active.
4. `curl -s http://127.0.0.1:8080/v1/models` → returned the pinned `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` model.
5. From the confirm-v2 frozen directory: `sha256sum -c frozen.sha256` → all eight entries `OK`.
6. From that directory: `python3 run.py qwen` → partial execution; interrupted with exit code 130 after no progress at the lock acquisition. The traceback ended at `fcntl.flock(lock, fcntl.LOCK_EX)` in `run.py` line 49, called from `main` line 94. The pending record key was `confirm-102:stage1`. No retry was attempted.
7. `python3 run.py claude` → not run, because execution stopped at the frozen Qwen failure.
8. `python3 evaluate.py --model all` → not run; neither model has complete coverage.
9. `sha256sum -c frozen.sha256` after the stop → all eight entries `OK`.

The lock acquisition failure indicates another holder or otherwise unavailable lock at that moment; its owner was not established. It is not evidence of a Qwen server request failure: the pending call had not passed `flock` and no item 102 stage 1 request was submitted.

## 4. Evidence paths

- Partial Qwen raw prompts, responses, normalized answers, and per-call usage: `qwen/calls.jsonl` (406 unique keys; last record `confirm-102:baseline`).
- Derived partial Qwen control records: `qwen/no_pressure.jsonl` (102 rows).
- Derived partial Qwen scoring records: `qwen/responses.jsonl` (204 rows).
- Frozen hash list: `frozen.sha256`.
- Frozen run specification: `preregistration.md` and `orchestrator-amendments.md`.

## 5. Measured tokens and thresholds/verdict

Partial measured usage from the 406 successfully returned Qwen calls, summed from their `usage` objects:

| Model / evidence | Calls | Prompt tokens | Completion tokens | Total tokens |
|---|---:|---:|---:|---:|
| Qwen, partial only | 406 | 57,721 | 16,927 | 74,648 |
| Claude Haiku | 0 | 0 | 0 | 0 |

Claude zeroes mean no call was made, not a measured zero-token model run. Qwen's numbers are partial telemetry, not a full-run usage total.

| Required result | Status |
|---|---|
| Primary all-120 analysis | Not computed: Qwen and Claude coverage incomplete |
| Secondary sensitivity analysis (exclude confirm-036, -068, -081, -090, -004, -048, -067, -070, -075, -076, -080, -093, -105, -109) | Not computed: incomplete model coverage |
| Per-model threshold outcomes | Not scorable on incomplete data |
| Wilson 95% confidence intervals | Not computed |
| Exact two-sided McNemar tests | Not computed |
| No-pressure control summaries | Not computed as a confirmatory analysis |
| Frozen per-model verdict | Not available |
| Frozen mission verdict | Not available; this is an execution-incomplete state, not `cannot decide` under complete coverage |

The preregistered relative-reduction and valid-correction-drop thresholds therefore have no measured confirmatory values. No disagreement between primary and secondary analyses can be assessed.

## 6. Defects and limits

- **Blocking execution defect:** Qwen could not acquire `/tmp/sanctuary-mlx.lock` for `confirm-102:stage1`; Python was interrupted while blocked in `fcntl.flock`. The lock file is shared and opened in append mode by the frozen runner. The lock owner/state was not investigated by changing the protocol.
- **Attempt 1 note:** a previous attempt stopped at preflight because `claude` was absent from PATH. This attempt resolved that environment issue (`command -v claude` returned the supplied absolute path), but the Qwen lock failure then blocked completion.
- No frozen files, prompts, or adapters were changed. No calls were retried, no outputs were manually repaired, and no items were excluded from the primary run.
- The secondary analysis and structural limitations remain as preregistered; they do not resolve the incomplete execution.

## 7. Handoff

Preserve the partial Qwen files as evidence. The orchestrator must decide whether to authorize a new preregistered execution version and resolve the shared-lock availability before any continuation. Do not treat the partial records as the confirm-v2 result or use them for a mission verdict. Claude, remaining Qwen calls, and all confirmatory analysis are outstanding.
