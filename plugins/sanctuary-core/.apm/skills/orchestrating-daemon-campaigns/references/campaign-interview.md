# Campaign interview

Plain-language intake for an unattended campaign. Ask only what the user has not already answered; offer the recommended option first; explain why a choice matters before showing the field.

## Contents
- [Phase 0: check the ground](#phase-0-check-the-ground) • [Phase 1: objective](#phase-1-objective) • [Phase 2: mode and steps](#phase-2-mode-and-steps) • [Phase 3: engine and rails](#phase-3-engine-and-rails) • [Phase 4: summary and approval](#phase-4-summary-and-approval) • [Phase 5: launch and report](#phase-5-launch-and-report) • [Example](#example)

## Phase 0: check the ground
Read-only, before asking anything:
```bash
C=plugins/sanctuary-core/.apm/skills/orchestrating-daemon-campaigns/scripts/campaign.py
python3 $C status                       # daemon_running, existing campaigns
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py board
python3 plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/usage_probe.py headroom
```
Report what you find in one or two lines (daemon off, a campaign already running, low quota). A daemon circuit that is BLOCKED_ON_HUMAN must be reviewed and reset first (`running-sanctuary-daemon`).

## Phase 1: objective
"What should run while you are away?"
- **Existing work**: named missions or backlog items. Read their manifests; propose one step per bounded deliverable.
- **A goal to break down**: propose 2-4 steps, each with one deliverable and one check. Show them and let the user edit.
- **A sweep**: "keep fixing until X passes" (tests, lint, a build).

Push back on open-ended goals ("improve the codebase"): the conductor is good at bounded, checkable steps.

## Phase 2: mode and steps
- **pipeline**: steps run in order; `depends_on` makes a step wait. If a step fails, everything that depends on it is BLOCKED and independent steps still run.
- **backlog**: independent steps, highest `priority` first. A step with `mission_id` and no `priority` uses that mission's `priority_score`.
- **sweep**: run the verify command; if it fails, run the fix step with the failure output in its prompt; repeat until it passes or the cycle limit is hit.

## Phase 3: engine and rails
Engine (one per campaign): shell (zero tokens), local Qwen (`llama`, local compute), Codex, Antigravity (`agy`), Claude (`claude` + model; Opus for hard architecture, Sonnet or Haiku to save quota). Never Copilot.

Rails, with defaults:
| Rail | Default | Why it matters |
|---|---|---|
| `max_cycles` | 5 | Total step runs, retries included: the real cost ceiling. |
| `step_timeout_sec` | 300 | A stuck run is killed (SIGTERM, then SIGKILL). |
| `max_consecutive_failures` | 2 | Stops the campaign (BLOCKED_ON_HUMAN) instead of burning quota on a broken plan. |
| `require_commit` | true | A step that changes nothing has not done its job. |
| `allowed_paths` | `**` | Globs a step may change; anything else fails the step. Narrow it. |
| `verify_cmd` | none | Run after each step (and as the sweep condition); failing it fails the step. |
| `revert_on_failure` | false | Hard-reset the worktree to the step's starting commit on failure. Task worktree only. |
| `max_retries_per_step` | 1 | One retry, with the failure reason in the prompt. |

Edits by claude or agy need `engine.allow_permission_bypass: true`: say plainly that the engine then runs tools and edits files without asking, bounded by the worktree, timeouts and breakers. Worst case to state: `max_cycles x step_timeout_sec` of run time, and up to `max_cycles` agent calls.

## Phase 4: summary and approval
```markdown
### Proposed campaign: <id>
- Goal: <one sentence>   Mode: <pipeline/backlog/sweep>   Engine: <engine + model>
- Steps: <s1 -> s2 (needs s1)> | <ranked list> | <verify `cmd`, fix: ...>
- Runs in: .worktrees/<task> (leased)
- Rails: <N> cycles max, <T> s per step, stop after <F> failures in a row, must commit: <y/n>,
  paths: <globs>, verify: `<cmd>`, revert and retry: <y/n>, edit permission: <isolated / bypass approved>
- Worst case: about <N x T / 60> min, up to <N> <engine> calls
Reply "go" to launch, or tell me what to change.
```
Write the manifest (see campaign-manifest.md), run `validate`, and show any errors before asking.

## Phase 5: launch and report
1. Lease and create the worktree if not done.
2. `python3 $C launch <manifest> [--start-daemon]` (`--start-daemon` only if the user agreed to start the daemon).
3. Report the conductor PID (and daemon PID), `status`, `tail -f ~/.sanctuary/daemon/campaigns/<id>/conductor.log`, and `python3 $C stop <id>`.
4. When it ends, read `status`, review the diff in the worktree, and continue the normal PR and closeout workflow.

## Example
> "Run a background campaign to clean up the API endpoints."

> Daemon is running and idle; Claude headroom 40%. Four things:
> 1. **Steps**: I suggest (a) add tests for `api/` endpoints, (b) refactor handlers, only if (a) completes, (c) update the API docs. OK?
> 2. **Engine**: Claude Sonnet to save quota, or Opus for the refactor?
> 3. **Rails**: 5 cycles, 10 min per step, stop after 2 failures, only `api/**` and `tests/**`, verify with `pytest -q tests/api`, revert and retry once?
> 4. **Edits**: Claude needs the permission-bypass opt-in to edit; with the worktree, paths and timeouts as the boundary, is that OK?
