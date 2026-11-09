---
name: running-sanctuary-daemon
description: "Runs the Sanctuary supervisor daemon: a zero-token, event-driven outer loop that dispatches authorized headless agent jobs (claude, codex, agy, local llama via cli-agent, or shell commands) with hard timeouts, a failure circuit breaker, policy gates and crash recovery. Use when planning or queueing unattended agent or test jobs (it interviews the user first: work type, pipeline steps, engine, safety rails), setting up a background run or a chained pipeline, starting or stopping the daemon, checking its status, or resetting it after a circuit-breaker trip. Never started by session bootstrap."
allowed-tools: Bash, Read
---

# Running the Sanctuary Daemon (`running-sanctuary-daemon`)

A deterministic supervisor that never calls a model. It waits on file events (macOS `kqueue`) for job files in `~/.sanctuary/daemon/queue/`, runs each authorized job in its own process group, and records the outcome. Model work happens only inside dispatched jobs, through the companion `cli-agent` (`run_agent.py`, `--isolated` by default).

## Contents
- [Constraints](#constraints) • [Plan a job (interactive)](#plan-a-job-interactive) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints
- **Never start it from bootstrap.** Session startup is observational. The human starts the daemon (foreground or a launchd agent); agents only enqueue, read status, and propose `reset`.
- **Human Gate.** Every job needs `authorized_by`. Enqueue only work the human approved in the current context.
- **Daemon states stay in the daemon.** QUEUED/RUNNING/COMPLETED/FAILED/TIMED_OUT/REJECTED and ACTIVE/BLOCKED_ON_HUMAN live in `~/.sanctuary/daemon/`, never in mission manifests.
- **No permission bypass by default.** cli-agent jobs run `--isolated`; a non-isolated job is rejected unless it sets `allow_permission_bypass: true`. `copilot` is refused (work repositories only).
- **Run jobs in a task worktree** (`--cwd .worktrees/<task>`), leased through `manage_coordination.py` as usual.

## Plan a job (interactive)
When the user wants something run in the background, do not enqueue straight away. Follow [plan-job-interview.md](references/plan-job-interview.md):
1. **Check the ground** (read-only): daemon running? circuit closed? leases?
2. **Ask four questions** in plain language: what kind of work, one run or a pipeline (`--after`), which engine (shell, local Qwen, Codex, Antigravity, Claude), which rails (time limit, must commit, worktree, edit permission).
3. **Show a plain-language summary** with the exact command(s) and their cost.
4. **Enqueue only after an explicit "go"**, then confirm with `status` and say how to watch the log.

Say plainly what the daemon cannot do (no retry-until-green loop; isolated claude/agy jobs cannot edit files unless the user approves `--allow-permission-bypass`).

## Quick Start
```bash
D=plugins/sanctuary-core/.apm/skills/running-sanctuary-daemon/scripts/sanctuary_daemon.py
python3 $D run                                   # human: start (foreground)
python3 $D enqueue --mission MISSION-0NN --authorized-by <approver> --timeout 60 --shell "make test"
python3 $D enqueue --mission MISSION-0NN --authorized-by <approver> --timeout 900 --require-commit \
  --cli claude --model claude-opus-5-5 --cwd .worktrees/<task> --instruction "<bounded task>"
python3 $D enqueue --mission MISSION-0NN --authorized-by <approver> --id step-2 --after step-1 --shell "make test"   # pipeline step
python3 $D status
python3 $D reset                                 # human, after reviewing a BLOCKED_ON_HUMAN trip
```

## Workflow
1. Human approves the work and starts the daemon (or it runs under launchd).
2. Agent leases the task and creates its worktree, then enqueues a bounded job with a timeout.
3. The daemon gates the job, dispatches it, enforces the timeout (SIGTERM, 2 s grace, SIGKILL) and records COMPLETED/FAILED/TIMED_OUT.
4. Three consecutive failures put the daemon in BLOCKED_ON_HUMAN; nothing more runs until `reset`.
5. Agent reads `status` and the job log, then continues the normal PR and closeout workflow.

## Verification
- `python3 scripts/test_daemon.py` runs the seven MISSION-001 criteria (detection < 2 s, SIGKILL timeout, idle zero-dispatch, circuit breaker, ordering and atomic writes, gates, crash recovery) plus C8 (`--after` pipelines) and C9 (`--allow-permission-bypass`) against throwaway homes. No model is called.
- `status` shows daemon state and every job; logs are in `~/.sanctuary/daemon/logs/`, events in `events.jsonl`.

## References
- Interactive planning interview: [plan-job-interview.md](references/plan-job-interview.md)
- Operations guide (start, launchd, rails, known gaps): [operations.md](references/operations.md)
- Sequence diagram: [sanctuary-daemon-sequence.mmd](../operating-sanctuary-guardian/references/sanctuary-daemon-sequence.mmd)
- Evidence and frozen protocol: `LEARNING/experiments/mission-001/`
