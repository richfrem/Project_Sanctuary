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
| `integrate` | Optional: `{"base": "main", "auto_merge_paths": ["LEARNING/research/<dossier>/**"]}`, see Integration. |

`bot.py validate <name>` checks all of it, including that the worktree exists and the engine is allowed.

## Schedules
The bot's own scheduler (inside the daemon, pure Python, any OS) checks about once a second while idle. Interval schedules are due `every` after the previous decision; `at` schedules at the next local `HH:MM`. Missed slots (computer off, bot stopped): `once` runs one catch-up run, `skip` jumps to the next slot. A run never starts while the previous run of the same automation is unfinished. `bot.py run-now <name>` runs an approved automation immediately.

## Approval
`bot.py approve <name> --by <approver>` validates the definition and stores its SHA-256 (canonical JSON) with the approver and time. Before every run the bot recomputes the hash; if the file changed, the automation shows `CHANGED_SINCE_APPROVAL` and does not run until approved again. The approver is passed to every job as `authorized_by`.

## Runs, memory and results
Each run gets an id `<name>-<YYYYMMDD-HHMMSS>` and goes through the conductor (a `job` is a one-step pipeline), so commit, path and verify checks apply to every run. The run's prompt starts with `PREVIOUS RUN OF THIS AUTOMATION:` (how the last run ended, its commit and files). When a run ends the bot appends to `<name>/history.jsonl` and `inbox.md`, and runs `notify_cmd` if configured (argv list, no shell; env `BOT_AUTOMATION`, `BOT_STATUS`, `BOT_SUMMARY`). A run whose conductor died without a final state is recorded as `CRASHED`.

## Integration
With an `integrate` block, every COMPLETED run is brought toward `origin/<base>` (level 2):
1. merge `origin/<base>` into the run's branch (a conflict aborts and is reported for review);
2. push the branch and open, or reuse, one PR;
3. **auto-merge only if** every changed file matches `auto_merge_paths` **and** none hits the built-in deny list, and no PR check is failing or pending;
4. fetch the merged revision and inspect its GitHub Actions runs and those at the companion repository's current remote main; pending, failed or unavailable checks keep delivery PARTIAL;
5. fast-forward the primary checkout only when it is clean and already on the base branch, then verify its HEAD matches the fetched remote revision; never switch another user's branch or overwrite edits;
6. fast-forward the runtime branch and invoke the Guardian's supported `merge_prep.py regenerate` there. A failed command or generated changes keep delivery PARTIAL, with evidence and the pending follow-up recorded in history and inbox. Generated exports need a separate scoped PR, never a direct main commit or a research-allowlist bypass.

No-change runs also perform steps 4–6, recovering already merged artifacts missing locally. The conductor's COMPLETED result alone does not make the automation COMPLETED: a configured integration must report `complete: true`. CI failures require inspection and repair; pending CI can be rechecked by a subsequent run. No workflow runs reported is recorded separately as `none`, not as passing tests.

The arXiv pipeline retains its digest → abstract-based relevance/impact/improvement analysis → future-mission seeds → save sequence. Seeds remain unreviewed candidates; the bot does not register or execute those missions.

The deny list is part of the bot (`scripts/integrate.py`) and a definition cannot override it: mission manifests and backlog files (`LEARNING/missions/**`, `index.jsonl`, `backlog.md`, `inbox.jsonl`), governance (`.agent/**`, `protocols/**`, `context/**`, AGENTS/CLAUDE/START_HERE/INSTALL/architecture, `apm.*`), `plugins/**`, `huggingface/**`, `.github/**`, and **any code file by extension**, even inside an allowed folder. `auto_merge_paths` must name a dossier under `LEARNING/research/`. Anything else stays an open PR for human review, and the inbox says why. The bot never syncs Hugging Face: that stays in session closeout.

## Pausing
`bot.py pause|resume <name>`. The bot pauses an automation itself after `pause_after_failed_runs` non-COMPLETED runs in a row; `resume` clears the count.

## Machine-local state
`~/.sanctuary/daemon/`: `config.json` (definitions dir, notify command), `automations/approvals.json`, `automations/state.json`, `automations/<name>/history.jsonl`, `inbox.md`, `campaigns/<run-id>/` (per-run state and logs), `queue/` and `logs/` (daemon jobs). `bot.py forget <name>` removes the approval and state, not the definition.
