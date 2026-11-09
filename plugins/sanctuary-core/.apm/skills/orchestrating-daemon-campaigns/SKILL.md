---
name: orchestrating-daemon-campaigns
description: "Interactively plans and launches on-demand daemon campaigns: multi-step autonomous bots, Grok-Bot-style background workflows and unattended campaigns (sequential pipelines, priority-ranked backlogs, or verify-and-fix sweeps) run by a zero-token conductor on top of the Sanctuary supervisor daemon, with triage, prompts built from the git trail, outcome checks, optional revert-and-retry and circuit breakers. Use when the user asks to set up an autonomous bot, an agent that works while they are away or overnight, a background worker for several tasks, a self-running pipeline, \"keep fixing until the tests pass\", or to launch, monitor or stop a campaign. A single one-off bot job belongs to running-sanctuary-daemon."
allowed-tools: Bash, Read, Write
---

# Orchestrating Daemon Campaigns (`orchestrating-daemon-campaigns`)

The brain on top of the daemon's muscle. The interview (here, in the main thread) shapes a campaign with the user; `scripts/campaign.py` runs it detached as a deterministic conductor that never calls a model. Each step goes to the `running-sanctuary-daemon` skill's daemon, which enforces gates, timeouts and process kill.

## Contents
- [Constraints](#constraints) • [Workflow](#workflow) • [Quick start](#quick-start) • [Verification](#verification) • [References](#references)

## Constraints
- **Interview first, launch last.** Never launch before the user approves the summary with an explicit "go" / "launch it". "Looks good" is not approval.
- **Human starts unattended work.** Launching is on demand from this conversation; session bootstrap never launches a campaign or the daemon.
- **Work in a leased task worktree** (`.worktrees/<task>`, via `manage_coordination.py checkout --create-worktree`). Revert-on-failure is refused anywhere else.
- **Cost is bounded by rails, not tokens.** Backends do not report usage reliably, so there is no token ceiling: max cycles, per-step timeout and the failure breaker are the limits. Say so.
- **No permission bypass unless approved.** Agent engines run isolated; editing with claude/agy needs `engine.allow_permission_bypass: true`, asked for explicitly.
- **Campaign states stay in `~/.sanctuary/daemon/campaigns/`**, never in mission manifests. The conductor does not push, open PRs or merge.

## Workflow
Follow [campaign-interview.md](references/campaign-interview.md):
1. **Check the ground** (read-only): daemon running? circuit closed? leases? headroom?
2. **Objective**: existing missions/backlog, a goal to decompose into steps, or a sweep.
3. **Mode and steps**: `pipeline` (depends_on), `backlog` (priority or a linked mission's `priority_score`), `sweep` (verify, then fix, until green).
4. **Engine and rails**: shell/local Qwen/Codex/Antigravity/Claude; max cycles (5), step timeout (300 s), breaker (2 failures), require commit, allowed paths, verify command, revert-and-retry.
5. **Summary**: plain-language plan, the manifest, worst-case run time and quota. Wait for "go".
6. **Launch and report**: PIDs, logs, `status` and `stop` commands.

## Quick start
```bash
C=plugins/sanctuary-core/.apm/skills/orchestrating-daemon-campaigns/scripts/campaign.py
python3 $C validate campaign.json
python3 $C launch campaign.json [--start-daemon]   # only after explicit approval
python3 $C status [<campaign-id>]
python3 $C stop <campaign-id>
```
Manifest format and an example: [campaign-manifest.md](references/campaign-manifest.md), [example-campaign.json](assets/example-campaign.json).

## Verification
- `python3 scripts/test_campaign.py`: 12 tests against a real daemon and real git worktrees, no model calls (pipeline order, blocked dependents, backlog ranking, prompt hand-off, revert-and-retry, allowed paths, commit check, breaker, sweep to green and exhaustion, validation, detached launch and stop, mission-priority triage).
- After a launch: `status` shows each step; `~/.sanctuary/daemon/campaigns/<id>/conductor.log` and the daemon's `logs/` hold the detail.

## References
- Interview: [campaign-interview.md](references/campaign-interview.md)
- Manifest and rails: [campaign-manifest.md](references/campaign-manifest.md)
- How the conductor decides (triage, prompts, evaluation, revert): [conductor-loop.md](references/conductor-loop.md)
- Runtime: the `running-sanctuary-daemon` skill (start, launchd, reset)
