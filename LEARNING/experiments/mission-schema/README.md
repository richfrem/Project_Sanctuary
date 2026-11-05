# Task: JSON Schema + validator for mission manifests, delegated to Claude Haiku 5.5

Pre-registered 2026-11-05 by claude-code, before dispatch. Ad-hoc task id `TASK-MISSION-SCHEMA` (not a backlog mission: astra/agy are allocating mission numbers).

Why: `LEARNING/research/missions/*.json` has no schema file. The only checks are inline in `manage_backlog.py validate` (title, hypothesis, falsification, scores, `projected_impact`). Field statistics over the 50 current manifests found things nothing checks, e.g. 6 manifests use verdict `CONFIRMED` while `record-result` accepts only CORROBORATED/FALSIFIED/ABANDONED.

Placement decision: schema beside the data (`LEARNING/research/mission.schema.json`, so any agent or bundle that reads the folder sees it); tooling in the skill that owns the backlog (`managing-sovereign-memory/scripts/validate_missions.py`, called by `manage_backlog.py validate`).

Dispatch: cli-agent skill, `--cli claude --model claude-haiku-5-5 --isolated` (no tools, text in/out). Haiku gets the field statistics, the current validator source and two sample manifests, and must return three files (schema, stdlib-only validator, tests). I extract and integrate them.

## Predictions (each can fail)
- P1. After mechanical extraction only, the schema + validator accept all 50 current manifests with 0 false rejections. Prior 55%.
- P2. Haiku's own tests pass on the first run. Prior 60%.
- P3. I need to fix at least one thing in its output (logic, not formatting). Prior 70%.
- P4. The dispatch costs < $0.30 and moves the Claude 5-hour window < 1 point. Prior 85%.
- P5. The finished validator rejects a deliberately broken manifest in each of 8 mutation tests I write myself (bad id, bad status, bad verdict, missing field, wrong type, bad related key, bad tier, score out of range). Prior 85%.

## Results
Dispatch: `run_agent.py --cli claude --model claude-haiku-5-5 --isolated`, 97 s wall. Measured from the delegate's own transcripts (separate project folder): **101,892 tokens = 4 in / 51,544 out / 17,580 cache-read / 50,344 cache-write (1h), $0.036**. The visible answer was about 21.8k characters (roughly 6k tokens), so around 45k of the output tokens were reasoning the caller never sees.

| Prediction | Result |
|---|---|
| P1 schema + validator accept all 50 manifests, 0 false rejections, after mechanical extraction | **Held**: `OK: 50 manifests` (35 tracked + agy's 15 untracked drafts, read-only) |
| P2 Haiku's tests pass first run | **Missed on the import path, held on logic**: the test imported the validator from its own folder, but I never told Haiku tests live in `tests/` beside `scripts/` (my spec gap). With the module importable, 14/14 passed; I changed one line (the `sys.path`) to fix it. |
| P3 I need at least one fix | **Held** (fixes, all listed below) |
| P4 cost < $0.30 and Claude window < 1 point | **Cost held ($0.036). Window not assessable**: the account-wide 5-hour window moved +4 and weekly +1 during the dispatch, but my own orchestrating session was running at the same time in another folder the probe cannot see. The record is therefore flagged `concurrent` and its window delta must not be used. |
| P5 8 mutations each rejected | **Held** 8/8, plus: legacy `CONFIRMED` accepted, unknown optional keys accepted |

Fixes I made to Haiku's output (it is not shipped as-is):
1. **Default missions folder was the script's own folder**, so the default run printed `OK: 0 manifests` and exited 0. My spec gave a default for the schema but not for the folder; still, a validator must never say OK when it checked nothing. Now: upward search for `LEARNING/research/missions`, and an empty folder is an error (exit 1).
2. Test import path (above).
3. Added a rule it was not asked for: the file name must equal the manifest `id` (all 50 already do).

Also changed, found while measuring the delegate: `usage_hook.record_delegated` gained a `claude` backend; its window starts at the pre-dispatch snapshot (not `now - seconds`); `--seconds` is only the recorded wall time; a claude delegate is flagged `concurrent` because it shares the account window with a claude orchestrator (this flag first failed to reach the ledger: the command line was built from a local variable, caught by a test that inspects the recorded command).

Wiring: `manage_backlog.py validate` now applies `mission.schema.json` whenever the root has one (so the pre-commit gate will start rejecting structurally invalid manifests, including future drafts by other agents). The 6 `CONFIRMED` verdicts are tolerated as legacy; `record-result` does not write that value.

Verdict on the placement question: schema beside the data in `LEARNING/research/`, validator in `managing-sovereign-memory/scripts/` (the skill that owns the backlog manager), documented in `references/agent-native-backlog.md`.
