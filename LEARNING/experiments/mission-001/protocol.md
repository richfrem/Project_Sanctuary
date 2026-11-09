# MISSION-001 protocol: zero-token supervisor daemon (frozen before the confirmatory run)

## Hypothesis (manifest, unchanged)
A zero-token, event-driven supervisor daemon can reliably orchestrate multi-step agent lifecycles, handle sequential state changes, and enforce operational rails (process timeouts, budget trips, state rollback) without burning tokens during idle state monitoring.

## Falsification criteria (manifest) made measurable
The hypothesis is FALSIFIED if, in the confirmatory run of `test_daemon.py --confirm`, any of these fails:

| ID | Criterion | Measure | Pass rule |
|---|---|---|---|
| C1 | Detect a queued job within 2 s | wall time from the atomic write of a QUEUED job to the daemon's RUNNING write | all 20 trials < 2.0 s |
| C2 | Hard timeout enforced by SIGKILL | job `sh -c 'sleep 60 & sleep 60; wait'` with `timeout_sec=2` | in all 5 trials: status TIMED_OUT, ended < timeout + 4 s (SIGTERM, 2 s grace, then SIGKILL), and no process of the job's process group is alive afterwards |
| C3 | No model inference while idle | daemon idles 15 s with an empty queue | 0 dispatches, and daemon CPU time < 1% of wall time |
| C4 | Failure rail (budget trip) | 3 consecutive jobs exiting non-zero | 3rd job leaves the daemon in BLOCKED_ON_HUMAN; a 4th QUEUED job is not dispatched until `reset` |
| C5 | Sequential state changes | 3 jobs queued in order | each runs exactly once, in queue order, states QUEUED -> RUNNING -> COMPLETED, no job file ever unparseable mid-run |
| C6 | Authorization and policy gates | job without `authorized_by`; job with `cli: copilot`; job with a forbidden flag | none is dispatched; each ends REJECTED with a reason |

CORROBORATED if all six pass. UNASSESSED if the suite cannot run (environment failure).

## Scope and instruments
- POC location: `LEARNING/experiments/mission-001/daemon/sanctuary_daemon.py` (stdlib only; macOS `select.kqueue` watcher, polling fallback).
- Tests use the `shell` runner with deterministic commands. No model is called by the confirmatory suite.
- The `cli-agent` runner (dispatching through `run_agent.py` with `--cwd/--allowed-root/--claim-dir/--timeout/--result-json`) is exercised once in a separate, labelled live demo in a scratch git repository. The demo is evidence of end-to-end wiring, not part of the verdict.
- Development: the suite is iterated in dev mode (fewer trials) until the instrument is correct; then protocol.md, the daemon and the test file are hash-frozen and committed, and `--confirm` runs once.

## Rollback (state rollback in the hypothesis)
"State rollback" is tested as: a job interrupted by daemon shutdown (SIGTERM) while RUNNING is returned to QUEUED on the next start (recovery), not lost or duplicated. This is criterion C7, assessed in the same suite:

| C7 | Crash recovery | daemon stopped while a job is RUNNING, then restarted | the job's process group is killed on stop; on restart the job is re-queued once (attempt 2) and completes |

## Limits
A deterministic fixture suite shows the daemon's mechanics on this machine. It does not show reliability of real agent work, behaviour under weeks of uptime, or Linux behaviour (inotify not tested). Detection latency is measured on a local APFS disk.

## Budget
Local CPU only for the suite (< 5 min). Live demo: one Claude Haiku dispatch through cli-agent, max 120 s.

## Development disclosures (recorded before the freeze; criteria unchanged)
- Dev run 1: C2 and C7 failed. Cause, a code defect: on macOS `killpg(pgid, 0)` raises EPERM when the only member left is the unreaped zombie group leader, which crashed the daemon inside `kill_group` (and left the C7 job RUNNING). Fix: reap the leader while waiting and treat EPERM as "only a zombie remains". Dev run 2: 7/7 pass.
- Found during design: cli-agent `run_agent.py` adds `--dangerously-skip-permissions` for claude and agy unless `--isolated` is given. The daemon now passes `--isolated` by default and rejects a non-isolated cli-agent job unless it sets `allow_permission_bypass: true`. C6 gained a fourth case for this (bypass without opt-in). Dev run 3: 7/7 pass.
