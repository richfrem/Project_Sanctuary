# MISSION-001: Zero-token supervisor daemon

**Verdict: CORROBORATED** on amended confirmatory run 2 (`freeze_v2.json`, commit `800a9bc9f`). Run 1 (`freeze.json`, commit `00a0cda4d`) failed C2 because of a test-design defect; the defect, the amendment and an operator mistake are disclosed in `protocol.md`.

## What was built
`sanctuary_daemon.py` (stdlib, about 400 lines): a kqueue file watcher on a job queue, an authorization and policy gate, an isolated process-group runner (shell commands, or agent CLIs through `cli-agent` `run_agent.py --isolated`), a hard SIGTERM/SIGKILL timeout, a 3-failure circuit breaker (BLOCKED_ON_HUMAN until `reset`), single-instance locking, atomic job writes and crash recovery. Supported copy: the `operating-sanctuary-bot` skill (Sanctuary Bot), which merged the original `running-sanctuary-daemon` skill on 2026-11-09. Sequence diagram: originally `operating-sanctuary-guardian/references/sanctuary-daemon-sequence.mmd`; moved and extended on 2026-11-09 to `plugins/sanctuary-core/.apm/skills/operating-sanctuary-bot/operating-sanctuary-bot-flow.mmd`.

## Results (`evidence/confirm_v2_results.json`)
| Criterion | Result |
|---|---|
| C1 detect a queued job < 2 s | 20/20; median 0.007 s, max 0.014 s |
| C2 hard timeout by SIGKILL | 5/5 TIMED_OUT at 2.12-2.13 s (2 s limit); process group gone each time |
| C3 no inference while idle | 15 s idle: 0 dispatches, 0.01 s CPU (0.07%) |
| C4 failure rail | 3 failures -> BLOCKED_ON_HUMAN; 4th job held; ran after `reset` |
| C5 sequential state changes | order 0,1,2; 3 dispatches; 0 unparseable reads during writes |
| C6 gates | no `authorized_by`, `copilot`, bypass flag, non-isolated without opt-in: all REJECTED, 0 dispatches |
| C7 crash recovery | job group killed on daemon stop, job back to QUEUED, completed on attempt 2 |

Run 1 (frozen design): C1, C3-C7 passed; C2 could not be measured (see protocol).

## Live demo (`evidence/live_demo/`; not part of the verdict)
One daemon pass (`run --once`, 2.5 s wall) ran a real Claude Haiku job through cli-agent (isolated; exit 0; output `SANCTUARY-DAEMON-OK`; usage not reported by the backend) and a shell job that made a verified commit (`require_commit`) in a scratch git repository.

## Findings
- **Code defect found and fixed in dev:** on macOS `killpg(pgid, 0)` raises EPERM when only the zombie group leader is left; reap the leader while waiting.
- **cli-agent gap:** `run_agent.py` adds `--dangerously-skip-permissions` for claude/agy unless `--isolated`. Isolated claude in `-p` mode cannot edit files, so unattended editing currently needs an explicit bypass opt-in. A narrower `--permission-mode acceptEdits` plus allowed-tools option belongs upstream in cli-agent.
- **Unattended auth:** Claude's sign-in warned of expiry within days; long-running use needs `claude setup-token`.
- **Rails interact:** the C2 test failure was the C4 rail working; tests of one rail must isolate it from the others.

## Bootstrap decision
The daemon is started by the human (foreground or launchd), never by session bootstrap: startup must stay observational, and starting a dispatcher is a state-changing act under the Human Gate. A read-only status line in the boot briefing is a possible follow-up.

## Limits
Mechanics only, on one macOS machine; one job at a time; no token-budget rail yet (wall-clock and failure-count only); Linux inotify untested; the demo used one cheap model and a trivial task, so it does not show the reliability of real agent work.
