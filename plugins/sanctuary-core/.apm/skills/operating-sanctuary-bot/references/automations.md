# Automations

A definition is `<definitions_dir>/<name>.json` (default `.agent/automations/`, set by `bot.py setup`). The file name equals `name`.

## Contents
- [Definition](#definition) • [Schedules](#schedules) • [Approval](#approval) • [Runs, memory and results](#runs-memory-and-results) • [Pausing](#pausing) • [Machine-local state](#machine-local-state)

## Definition
| Field | Meaning |
|---|---|
| `name`, `description` | Identity; `name` must equal the file name. |
| `schedule` | `{"manual": true}` or `{"every": "30m"\|"6h"\|"1d"\|"1w", "at": "HH:MM" (whole days only), "catch_up": "once"\|"skip"}`. |
| `worktree` | Task worktree, relative to the repository root or absolute. |
| `engine` | `{"cli": "shell"\|"llama"\|"codex"\|"agy"\|"claude", "model": ..., "allow_permission_bypass": false}`. |
| `job` | One step: `{"goal": "..."}` (AI) or `{"shell": "..."}`; optional `verify`. |
| `mode`, `steps`, `sweep` | Several steps instead of `job` (see campaign-manifest.md). |
| `rails` | Per-run limits (see campaign-manifest.md). |
| `pause_after_failed_runs` | Default 3. |
| `mission_id` | Optional default mission id for the run's jobs. |

`bot.py validate <name>` checks all of it, including that the worktree exists and the engine is allowed.

## Schedules
The bot's own scheduler (inside the daemon, pure Python, any OS) checks about once a second while idle. Interval schedules are due `every` after the previous decision; `at` schedules at the next local `HH:MM`. Missed slots (computer off, bot stopped): `once` runs one catch-up run, `skip` jumps to the next slot. A run never starts while the previous run of the same automation is unfinished. `bot.py run-now <name>` runs an approved automation immediately.

## Approval
`bot.py approve <name> --by <approver>` validates the definition and stores its SHA-256 (canonical JSON) with the approver and time. Before every run the bot recomputes the hash; if the file changed, the automation shows `CHANGED_SINCE_APPROVAL` and does not run until approved again. The approver is passed to every job as `authorized_by`.

## Runs, memory and results
Each run gets an id `<name>-<YYYYMMDD-HHMMSS>` and goes through the conductor (a `job` is a one-step pipeline), so commit, path and verify checks apply to every run. The run's prompt starts with `PREVIOUS RUN OF THIS AUTOMATION:` (how the last run ended, its commit and files). When a run ends the bot appends to `<name>/history.jsonl` and `inbox.md`, and runs `notify_cmd` if configured (argv list, no shell; env `BOT_AUTOMATION`, `BOT_STATUS`, `BOT_SUMMARY`). A run whose conductor died without a final state is recorded as `CRASHED`.

## Pausing
`bot.py pause|resume <name>`. The bot pauses an automation itself after `pause_after_failed_runs` non-COMPLETED runs in a row; `resume` clears the count.

## Machine-local state
`~/.sanctuary/daemon/`: `config.json` (definitions dir, notify command), `automations/approvals.json`, `automations/state.json`, `automations/<name>/history.jsonl`, `inbox.md`, `campaigns/<run-id>/` (per-run state and logs), `queue/` and `logs/` (daemon jobs). `bot.py forget <name>` removes the approval and state, not the definition.
