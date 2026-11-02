# Pre-Committed Rollback Contract

> Fill in the thresholds below BEFORE starting any tool-smithing loop.
> The agent uses these to determine when to abort automatically.

## Rollback Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Consecutive crashes | >= 3 in a row | Halt loop. Present findings. Wait for human input. |
| Toolbox regression | Any committed tool now fails its own `--test` block | Revert that commit via `git revert`. Log `regression` entry. |
| Sandbox violation | Any constraint in safety-sandbox.md broken | Delete candidate. Log `SANDBOX_VIOLATION`. Continue. |
| Loop budget exceeded | > `max-iterations` (default: 10) | Halt. Summarize ledger. Present top gap still open. |
| Disk growth | `scripts/toolbox/` exceeds 50 MB | Halt. Run `clean-memory-logs.py` if available, else alert human. |

## Rollback Procedure

1. `git revert HEAD` (if a bad commit was made)
2. Restore `evals/results.tsv` from `evals/results.tsv.bak` (agent creates backup before each cycle)
3. Log the rollback in `evals/results.tsv` with decision `rollback` and reason.
4. Resume from the last `keep` state.

## Human Gate Conditions (Graduated Autonomy)

The following changes REQUIRE explicit human approval before commit:
- Any script that modifies or deletes an existing tool in `scripts/toolbox/`
- Any addition to `scripts/toolbox/requirements.txt` (new external dependency)
- Any change to the `program.md` strategy goal
- Any script > 100 lines
