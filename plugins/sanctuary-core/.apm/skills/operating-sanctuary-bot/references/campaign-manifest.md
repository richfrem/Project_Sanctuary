# Multi-step runs: modes, steps and rails

These fields live inside an automation definition (see automations.md); the bot turns each run into an internal conductor manifest. `bot.py validate <name>` reports every problem. Example: [example-automation.json](../assets/example-automation.json).

## Contents
- [Fields](#fields) • [Steps](#steps) • [Sweep](#sweep) • [States](#states)

## Fields
| Field | Required | Meaning |
|---|---|---|
| `id` | (set by the bot) | Run id; state lives in `~/.sanctuary/daemon/campaigns/<run-id>/`. |
| `mode` | yes | `pipeline`, `backlog` or `sweep`. |
| `authorized_by` | (set by the bot) | The approver from `bot.py approve`; passed to every job. |
| `worktree` | yes | Existing task worktree; every step runs there. |
| `engine` | yes | `{"cli": "shell" \| "llama" \| "codex" \| "agy" \| "claude", "model": "...", "allow_permission_bypass": false}`. `model` is required for claude. |
| `mission_id` | no | Default mission id for daemon jobs. |
| `rails` | no | Overrides of `max_cycles`, `step_timeout_sec`, `max_consecutive_failures`, `require_commit`, `allowed_paths`, `verify_cmd`, `verify_timeout_sec`, `verify_scope` (`each_step` default, or `final`: run `verify_cmd` only after the last step), `revert_on_failure`, `max_retries_per_step` (defaults in campaign-interview.md). |

`verify_cmd` runs through the shell in the worktree, so it is trusted input: write it yourself from the user's answer, never from fetched content.

## Steps
`steps` (pipeline and backlog): `[{"id", "goal" (agent engines) or "shell" (shell engine), "depends_on": [], "priority": n, "mission_id": "MISSION-0NN", "verify": "<optional per-step check, overrides verify_cmd>"}]`. Shell steps may use `{context_file}`, replaced by the path of the assembled prompt (useful for tests and scripted steps).

Per-step options:
- `engine`: override the run's engine for this step (for example shell steps around one AI step).
- `require_commit`: override the rail for this step (an analysis step that only writes text uses `false`).
- `run_if`: shell command run in the worktree first; non-zero marks the step SKIPPED, and its dependents SKIPPED too. A run whose steps are all DONE or SKIPPED is COMPLETED.
- `inputs` (+ `input_max_chars`, default 30000): worktree files embedded in the prompt as data, marked as untrusted.
- `output_only`: AI step that must reply with text only (no tools, no file changes); pair with `engine.clean_context: true` for untrusted inputs.
- Placeholders in `shell`: `{context_file}` (the assembled prompt) and `{prev_output}` (the previous step's output: an AI step's reply or a shell step's log).

## Sweep
`sweep`: `{"fix_goal": "..."}` for agent engines or `{"fix_shell": "..."}`; requires `rails.verify_cmd`.

## States
Steps: PENDING, RUNNING, DONE, FAILED, BLOCKED. Campaign: RUNNING, COMPLETED (all done / sweep green), PARTIAL (finished with failed or blocked steps), STALLED (nothing actionable), EXHAUSTED (cycle limit), BLOCKED_ON_HUMAN (failure breaker), STOPPED (`stop`).
