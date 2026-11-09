# How the conductor decides

`scripts/conductor.py` (`Conductor.run`, internal; started by the bot for each run) with selection in `scripts/triage.py`. One step at a time; zero tokens between steps (it sleeps while polling the daemon job file).

## Contents
- [Each cycle](#each-cycle) • [Prompt hand-off](#prompt-hand-off) • [Evaluation](#evaluation) • [Failure handling](#failure-handling) • [Limits](#limits)

## Each cycle
1. Stop if `stop` was requested, the campaign is blocked, or `cycles >= max_cycles`.
2. **Triage**: mark steps whose dependency FAILED/BLOCKED as BLOCKED; among steps whose dependencies are DONE, pick the first (pipeline) or the highest score (backlog: `priority`, else the linked mission's `priority_score`, else 0). Sweep: run `verify_cmd`; green ends the campaign.
3. Record the worktree HEAD, write the prompt to `<job>.context.txt`, enqueue one daemon job (`--timeout step_timeout_sec`, `--cwd` the worktree, engine flags) and wait for its terminal status.
4. Evaluate, record history, update counters.

## Prompt hand-off
Each prompt carries: `PREVIOUS STEP:` (last successful step's commit subject and hash, files changed, verify result), `RETRY:` (on a retry, the failure reason and whether it was reverted), `GOAL:` and `RULES:` (stay in the worktree, allowed paths, make verify pass, commit).

## Evaluation
A step passes only if all hold: daemon job COMPLETED (not FAILED, TIMED_OUT or REJECTED); a new commit exists when `require_commit`; every changed file matches `allowed_paths`; the step's check exits 0: its own `verify` if set, otherwise `verify_cmd` (every step, or only the last one when `verify_scope` is `final`). An exit code of 0 alone is not success.

## Failure handling
- `revert_on_failure`: `git reset --hard <start>` and `git clean -fd`, only in a `.worktrees/` checkout on a non-main branch (re-checked at revert time).
- Retry up to `max_retries_per_step` with the reason in the prompt; then FAILED, which blocks dependents.
- `max_consecutive_failures` failures in a row: BLOCKED_ON_HUMAN. The daemon has its own breaker (3) as a second stop.

## Limits
No token or cost ceiling (usage is not reported reliably); no parallel steps; one engine per campaign; `stop` lets the current daemon job finish or time out. Mechanics are tested with shell steps; agent-step quality depends on the engine and the instructions.
