# MISSION-035 experiment record (pre-registration, written before building)

Lease: MISSION-035 / claude-code, worktree `.worktrees/claude-code-mission-035`. Background: `research-notes.md`.

## Phase predictions (P1-P4, build and verify)
| # | Prediction | Falsified if |
|---|---|---|
| P1a | `claude -p /usage`, `agy -p /usage --output-format json` each cost 0 tokens and parse into window records (claude: session% + week% with resets; agy: >= 5 buckets, each with window, remaining fraction, reset time) | any costs > 0 tokens, or parse yields < those fields on the real captured output |
| P1b | Codex windows are readable from the newest rollout that has a `token_count` event (not simply the newest file, since TUI-only sessions have none) and carry primary 300-min and secondary 10080-min used_percent | newest-file reading returns no data while an older file has it and the scan-back rule does not fix it |
| P2 | Zero secret leakage: the probe's output and ledger records pass a credential-pattern scan, and the probe never opens `auth.json` | scan finds a token/key pattern, or auth.json is opened |
| P3 | Concurrent-activity flag: a snapshot interval during which a second Claude transcript is written is flagged concurrent; an interval with only one active transcript is not | flag misclassifies in the synthetic test set |
| P4 | Measured records round-trip through `track_budget.py record --source measured` into `events.jsonl` without breaking existing `status`/`check` output | existing tests fail or status output changes for old records |

## Real-mission collection predictions (replaces the synthetic pilot, per the human's direction 2026-11-04)
Data is gathered automatically at checkout/checkin of real missions; no tasks are run for the experiment's sake. The verdict is deferred until >= 12 real measured tasks exist.
- Per-task token deltas agree with each vendor's reported total within +/-10% for >= 3 of 4 backends.
- Held-out MAPE of the per-class median predictor <= 30% and >= 20% lower than the static per-task ceiling (a ceiling, not a point estimate, so this baseline is generous to the predictor).
- Effort multiplier: within a task class, models differ measurably in cost/time/retries; if not, report the multiplier as noise.
- Usable (non-noisy) window-delta fraction is reported, no target.
- Stop rule: the collection is passive and spends no extra quota, but the recommender surfaces any window under 15% remaining.
