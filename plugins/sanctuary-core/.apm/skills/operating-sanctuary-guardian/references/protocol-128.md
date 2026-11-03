# Protocol 128: The Learning Loop & Guardian Lifecycle

The authoritative definition is [`protocols/128_Hardened_Learning_Loop_v3.0.md`](../../../../../../protocols/128_Hardened_Learning_Loop_v3.0.md) (v3.1). This file is a derived quick reference; if it disagrees with the protocol, the protocol wins.

## Five stages and the guardian commands for each

| Stage | Purpose | Commands and scripts (this skill unless noted) |
|---|---|---|
| 1. Boot Orientation | Inherit state, pick the work, check for other agents | `guardian_wakeup.py --mode HOLISTIC`; `forge_llm.py status` (local model gate); `manage_coordination.py board` then `checkout --create-worktree`; `managing-sovereign-memory/disclose.py` |
| 2. Sovereign Execution | Do the work under the Human Gate | installed APM skills; `forge_llm.py loop` for bounded local agent runs; `evaluating-agent-work/track_budget.py` |
| 3. Retrospective | Keep insights even when code rolls back | `learning_debrief.py`; `managing-sovereign-memory/manage_backlog.py record-result` and `compile`; `record_insight.py` |
| 4. Technical Seal | Verify and stamp | skill test suites; pre-commit cognitive gate; `score_contribution.py`; `chronicle_manager.py create`; `capture_snapshot.py --type seal` |
| 5. Soul Persistence | Survive a local reset | `preserving-soul-traces/trace_extractor.py --append` then `--verify`; `sync_soul_dataset.py`; `managing-huggingface-assets` for single files |

## Mapping from the former "8 phases"

Earlier versions of this file listed eight phases. They map onto the five stages: Phase I -> Stage 1; Phases II-IV (formulation, execution, inner loop and review) -> Stage 2, with the pre-completion checks in Stage 4; Phase V -> Stage 3; Phase VI -> Stage 4; Phases VII-VIII (persistence, closure) -> Stage 5.

## Notes

- `persist_soul.py` uploads one sealed snapshot via the global `managing-huggingface-assets` skill (`--dry-run` first); for the whole dataset use `trace_extractor.py` plus `sync_soul_dataset.py`. It was repaired on 2026-11-03.
