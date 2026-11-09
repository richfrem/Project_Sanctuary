# Running Sanctuary Bot

A zero-token outer loop for headless agent runs. It waits on file events for job files, runs each authorized job in its own process group, and enforces timeouts, a failure circuit breaker, policy gates and crash recovery. It never calls a model; models run only inside the jobs it dispatches. Flow diagram: [`operating-sanctuary-bot-flow.mmd`](../operating-sanctuary-bot-flow.mmd).

Status: proof of concept, corroborated by MISSION-001 (`LEARNING/experiments/mission-001/`). Stdlib only; macOS `kqueue` watcher with a polling fallback.

## Start and stop

Users: `bot.py start`, `bot.py stop`, `bot.py status`. The lower-level daemon commands below are for debugging.

### Daemon (internal, foreground)

```bash
D=plugins/sanctuary-core/.apm/skills/operating-sanctuary-bot/scripts/sanctuary_daemon.py   # internal; prefer bot.py
python3 $D run                       # Ctrl-C or SIGTERM to stop; state in ~/.sanctuary/daemon/
python3 $D status                    # daemon state and jobs (from another terminal)
```

Queue work (the enqueue is the authorization record; leave `--authorized-by` empty and the job is rejected):

```bash
# deterministic job
python3 $D enqueue --mission MISSION-001 --authorized-by <approver> --timeout 60 --shell "make test"

# agent job through cli-agent (isolated by default: no permission bypass)
python3 $D enqueue --mission MISSION-0NN --authorized-by <approver> --timeout 900 \
  --cli claude --model claude-opus-5-5 --cwd .worktrees/<task> \
  --instruction "Implement X, run the tests, commit with message 'feat: X'." --require-commit
```

Pipelines: `--after <job-id>` makes a job wait for another; it runs only if that job COMPLETED and is otherwise marked SKIPPED. Edits by claude or agy: add `--allow-permission-bypass` (cli-agent jobs only; requires explicit human approval), since isolated `-p` runs cannot edit files.

After 3 consecutive failures the daemon stops dispatching (`BLOCKED_ON_HUMAN`). Review the logs in `~/.sanctuary/daemon/logs/`, then `python3 $D reset`.

## Should session bootstrap start it?

**No.** Start it yourself, or opt in to launchd. Three reasons:

1. **Startup is observational.** Sanctuary's rules keep session startup read-only: it reports state and never starts state-changing work. A daemon that can launch agents which edit and commit is state-changing.
2. **Human Gate.** A bootstrap that starts the dispatcher would let any agent session turn on unattended execution. The daemon should run because you chose to run it.
3. **Lifetime.** A daemon should outlive any one agent session. Tying it to bootstrap would start duplicates (the lock refuses a second one) and stop it when a session ends.

A good later addition (not done here): the boot briefing reports the daemon read-only, for example "daemon: ACTIVE, 2 queued, circuit closed" from `status`.

### Optional: start at login

The bot schedules everything itself; only starting it at login is OS-specific. The command is always `python3 <skill>/scripts/sanctuary_daemon.py run`.

- **Linux (systemd user unit)** `~/.config/systemd/user/sanctuary-bot.service`: `[Service]` `ExecStart=/usr/bin/python3 /home/<you>/.../operating-sanctuary-bot/scripts/sanctuary_daemon.py run` `Restart=on-failure`, `[Install]` `WantedBy=default.target`; then `systemctl --user enable --now sanctuary-bot`.
- **Windows (Task Scheduler)**: `schtasks /Create /SC ONLOGON /TN SanctuaryBot /TR "py C:\\Users\\<you>\\...\\operating-sanctuary-bot\\scripts\\sanctuary_daemon.py run"` (process groups and SIGKILL differ on Windows: untested there).
- **macOS (launchd)**: below.

#### macOS launchd

Save as `~/Library/LaunchAgents/org.sanctuary.daemon.plist`, replace `<you>` with your user name and adjust paths, then `launchctl load ~/Library/LaunchAgents/org.sanctuary.daemon.plist`. Unload to stop.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>org.sanctuary.daemon</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/python3</string>
    <string>/Users/<you>/Projects/Project_Sanctuary/plugins/sanctuary-core/.apm/skills/operating-sanctuary-bot/scripts/sanctuary_daemon.py</string>
    <string>run</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/<you>/.sanctuary/daemon/launchd.log</string>
  <key>StandardErrorPath</key><string>/Users/<you>/.sanctuary/daemon/launchd.log</string>
</dict></plist>
```

## Rails and what enforces them

| Rail | Enforced by |
|---|---|
| Hard timeout | daemon: SIGTERM to the job's process group, 2 s grace, SIGKILL (and run_agent's own `--timeout`) |
| Failure ceiling | daemon: 3 consecutive FAILED/TIMED_OUT -> BLOCKED_ON_HUMAN until `reset` |
| Authorization | daemon: job needs `authorized_by` |
| Backend policy | daemon: `copilot` refused; bypass flags refused; cli-agent runs `--isolated` unless the job sets `allow_permission_bypass` |
| Worktree boundary, duplicate launch | run_agent `--cwd/--allowed-root`, `--claim-dir/--unit` |
| One daemon at a time | `flock` on `daemon.lock` |
| Crash recovery | on start, jobs left RUNNING are killed and re-queued (max 2 attempts) |

## Known gaps

- `run_agent.py` without `--isolated` adds `--dangerously-skip-permissions` for claude and agy. Isolated claude runs get no edit permission in `-p` mode, so an unattended **editing** run currently needs the explicit bypass opt-in (`enqueue --allow-permission-bypass`).
- No retry-until-green loop; use a fixed `--after` pipeline. A narrower option (`--permission-mode acceptEdits` plus an allowed-tools list) belongs in the companion `cli-agent` skill.
- Claude sign-in tokens expire (the live demo warned of expiry within days); for unattended runs use `claude setup-token`.
- No token budget per job yet: only wall-clock and failure-count rails. A cost rail could read run_agent's `result.json` usage.
- Linux (`inotify`) is untested; the polling fallback is used there.
- One job at a time; no priorities beyond queue order.

## Tests

`python3 scripts/test_daemon.py` (runtime), `scripts/test_campaign.py` (multi-step), `scripts/test_bot.py` (automations) (quick) or `python3 scripts/test_daemon.py --confirm --evidence-dir <dir>` (MISSION-001 trial counts). Protocol and results: `LEARNING/experiments/mission-001/`.
