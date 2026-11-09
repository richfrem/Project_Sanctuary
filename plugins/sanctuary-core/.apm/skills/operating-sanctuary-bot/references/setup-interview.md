# Setting up an automation: interview

Plain-language intake. Ask only what is still unknown, recommend an option first, and explain why a choice matters before naming a field. The user should never need to know the words daemon, conductor or manifest.

## Contents
- [0. Check the plumbing](#0-check-the-plumbing) • [1. What should it do](#1-what-should-it-do) • [2. When](#2-when) • [3. One step or several](#3-one-step-or-several) • [4. Engine](#4-engine) • [5. Safety rails](#5-safety-rails) • [6. Results](#6-results) • [7. Summary and approval](#7-summary-and-approval) • [Example](#example)

## 0. Check the plumbing
Run `bot.py doctor` (read-only). Not set up: run `bot.py setup` and say so in one line. Bot not running: mention it; offer to start it at the end. An automation that is paused after failures: show its last result and ask whether to fix or resume it first. Check headroom for AI engines (`evaluating-agent-work/scripts/usage_probe.py headroom`).

## 1. What should it do?
Get one concrete, checkable outcome per step ("run the test suite and fix failures in `src/`", not "improve the code"). If it reads or changes code, it needs a task worktree: create and lease one (`manage_coordination.py checkout --mission <id> --agent <agent> --create-worktree`).

## 2. When?
- **On demand only** (`"schedule": {"manual": true}`), run with `bot.py run-now`.
- **Every N minutes/hours/days** (`{"every": "6h"}`), or **daily/weekly at a time** (`{"every": "1d", "at": "07:00"}`).
- **If the computer was off when it was due:** run once on return (`catch_up: once`, default) or skip missed runs (`skip`).
- A run never starts while the previous one is still going.

## 3. One step or several?
- **One step** (`job`): one instruction, checked, done.
- **Several steps in order** (`mode: pipeline`, `depends_on`): a failed step blocks the steps that need it; others continue.
- **A ranked list** (`mode: backlog`): highest priority first, or a linked mission's priority score.
- **Keep fixing until a check passes** (`mode: sweep`): check, fix, repeat until green or out of cycles.

## 4. Engine
Shell (no AI, zero tokens), local Qwen (`llama`, local compute), Codex, Antigravity (`agy`), Claude (Opus for hard work; Sonnet or Haiku to save quota). Never Copilot. Editing files with claude or agy needs `engine.allow_permission_bypass: true`: say plainly that the engine then edits and runs tools without asking, bounded by the worktree, allowed paths, timeouts and failure limits.

## 5. Safety rails
Recommend and explain: time limit per step (`step_timeout_sec`, default 300), total step runs per run (`max_cycles`, 5), must commit (`require_commit`), allowed paths (`allowed_paths`), a check command (`verify_cmd`, and `verify_scope: final` when it only makes sense after the last step), revert and retry once on failure (`revert_on_failure`, worktree only), stop a run after 2 failures in a row (`max_consecutive_failures`), and pause the automation after 3 failed runs (`pause_after_failed_runs`). Worst case to state: `max_cycles x step_timeout_sec` per run, times runs per day.

## 6. Results
Every run is recorded in `~/.sanctuary/daemon/inbox.md` and `bot.py history <name>`, and the next run is told how the previous one ended. Optional: a notification command (`bot.py setup --notify-cmd "<command>"`; it receives `BOT_AUTOMATION`, `BOT_STATUS`, `BOT_SUMMARY`).

## 7. Summary and approval
```markdown
### Proposed automation: <name>
- Does: <one sentence>        When: <manual / every 6h / daily 07:00> (missed runs: <once/skip>)
- Steps: <one step | a -> b | ranked list | fix until `<check>` passes>
- Engine: <engine + model>    Runs in: .worktrees/<task>
- Rails: <T> s per step, <N> steps per run, must commit <y/n>, paths <globs>, check `<cmd>`, revert and retry <y/n>,
  edit permission <isolated / bypass approved>, pauses itself after <K> failed runs
- Worst case: ~<N x T / 60> min and up to <N> <engine> calls per run
Reply "go" to save and approve it, or tell me what to change.
```
On "go": write `.agent/automations/<name>.json`, `bot.py validate`, commit it on a feature branch (PR per the normal workflow), `bot.py approve <name> --by <approver>`, start the bot if needed, then show `bot.py status`.

## Example
> "Set up a bot that runs the tests every night and fixes what breaks."

> The bot is set up but not running; Claude headroom 40%. A few choices:
> 1. **When:** every night at 02:00, skip nights the laptop was off?
> 2. **Steps:** run `pytest -q`; if it fails, have Claude Sonnet fix it, repeating up to 3 times?
> 3. **Where:** a dedicated `.worktrees/nightly-fixes` branch, changes limited to `src/**` and `tests/**`, each step stopped after 10 minutes, reverted if the fix makes things worse?
> 4. **Edits:** Claude needs the edit-permission opt-in for this; with the worktree, paths and time limits as the boundary, OK?
> 5. **Results:** in your bot inbox each morning; want a desktop notification as well?
