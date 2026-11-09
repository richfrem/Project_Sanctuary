# Campaign manifest

JSON file passed to `campaign.py validate|launch`. `validate` reports every problem; `launch` refuses an invalid manifest. Example: [example-campaign.json](../assets/example-campaign.json).

## Contents
- [Fields](#fields) • [Steps](#steps) • [Sweep](#sweep) • [States](#states)

## Fields
| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Campaign id; state lives in `~/.sanctuary/daemon/campaigns/<id>/`. |
| `mode` | yes | `pipeline`, `backlog` or `sweep`. |
| `authorized_by` | yes | Who approved it; passed to every daemon job (the daemon rejects jobs without it). |
| `worktree` | yes | Existing task worktree; every step runs there. |
| `engine` | yes | `{"cli": "shell" \| "llama" \| "codex" \| "agy" \| "claude", "model": "...", "allow_permission_bypass": false}`. `model` is required for claude. |
| `mission_id` | no | Default mission id for daemon jobs. |
| `rails` | no | Overrides of `max_cycles`, `step_timeout_sec`, `max_consecutive_failures`, `require_commit`, `allowed_paths`, `verify_cmd`, `verify_timeout_sec`, `verify_scope` (`each_step` default, or `final`: run `verify_cmd` only after the last step), `revert_on_failure`, `max_retries_per_step` (defaults in campaign-interview.md). |

`verify_cmd` runs through the shell in the worktree, so it is trusted input: write it yourself from the user's answer, never from fetched content.

## Steps
`steps` (pipeline and backlog): `[{"id", "goal" (agent engines) or "shell" (shell engine), "depends_on": [], "priority": n, "mission_id": "MISSION-0NN", "verify": "<optional per-step check, overrides verify_cmd>"}]`. Shell steps may use `{context_file}`, replaced by the path of the assembled prompt (useful for tests and scripted steps).

## Sweep
`sweep`: `{"fix_goal": "..."}` for agent engines or `{"fix_shell": "..."}`; requires `rails.verify_cmd`.

## States
Steps: PENDING, RUNNING, DONE, FAILED, BLOCKED. Campaign: RUNNING, COMPLETED (all done / sweep green), PARTIAL (finished with failed or blocked steps), STALLED (nothing actionable), EXHAUSTED (cycle limit), BLOCKED_ON_HUMAN (failure breaker), STOPPED (`stop`).
