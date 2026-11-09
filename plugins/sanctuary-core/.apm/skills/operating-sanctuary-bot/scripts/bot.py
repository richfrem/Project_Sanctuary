#!/usr/bin/env python3
"""bot.py - the one interface to Sanctuary Bot.

Sanctuary Bot = runtime daemon (sanctuary_daemon.py) + scheduler (automations.py) +
multi-step conductor (conductor.py). Users and agents only need this file.

  bot.py setup [--definitions DIR] [--notify-cmd "cmd args"]   configure once (idempotent)
  bot.py doctor                                                check host, config, daemon, engines
  bot.py start | stop                                          start (detached) / stop the bot
  bot.py status                                                bot, automations, recent results
  bot.py list | show NAME | validate NAME | history NAME
  bot.py approve NAME --by APPROVER                            approve this exact definition (content hash)
  bot.py pause NAME | resume NAME | run-now NAME | forget NAME
  bot.py run FILE                                              one-off run of a definition, without scheduling
  bot.py reset                                                 close the daemon circuit breaker after review
State: ~/.sanctuary/daemon (override with --home or SANCTUARY_DAEMON_HOME).

Key Input Dependencies:
- ~/.sanctuary/daemon/config.json (written by `setup`; definitions folder, notify command)
- <definitions_dir>/<name>.json automation definitions (default: repo .agent/automations/)
- ~/.sanctuary/daemon/state.json, automations/approvals.json, automations/state.json, inbox.md
- Sibling scripts: automations.py, conductor.py, sanctuary_daemon.py, verify_health.py
- Env: SANCTUARY_DAEMON_HOME (state folder override)

Key Functions:
- repo_root(): The main checkout's root (also from inside a worktree), or None outside git.
- config(): The bot's config, or None when not set up.
- need_config(): The bot's config, or exit with a hint to run `bot.py setup`.
- registry(): The automation registry for the configured definitions folder.
- when(): Format an epoch timestamp as local `YYYY-MM-DD HH:MM`, or '-' when empty.
- daemon_pid(): The running daemon's pid, or None.
- cmd_setup(): Configure the definitions folder (default: repo .agent/automations) and notify command, then run doctor.
- cmd_doctor(): Print host, config, bot-process and engine checks; non-zero if host or config is missing.
- cmd_start(): Start the daemon detached if it is not running.
- cmd_stop(): Stop the daemon (SIGTERM); a running job is killed and re-queued once.
- rows(): One status row per automation: state, schedule, last run, next due time.
- cmd_list(): Print the automations table.
- cmd_status(): Print bot state, the automations table and the latest inbox lines.
- cmd_show(): Print one automation's definition, approval and state as JSON.
- cmd_validate(): Validate one definition and print its problems.
- cmd_approve(): Approve a definition for the named approver (content hash).
- cmd_simple(): pause, resume, run-now or forget one automation.
- cmd_history(): Print an automation's finished runs.
- cmd_run(): One-off run of a definition file (job or multi-step) through the conductor, no schedule.
- cmd_reset(): Close the daemon's global circuit breaker.
- main(): Parse arguments and dispatch to the command handlers.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import automations as A  # noqa: E402
import conductor as C  # noqa: E402

DAEMON = HERE / "sanctuary_daemon.py"
HOME = Path(os.environ.get("SANCTUARY_DAEMON_HOME", Path.home() / ".sanctuary" / "daemon"))


# The main checkout's root (also from inside a worktree), or None outside git.
def repo_root() -> Path | None:
    """The main checkout's root (also from inside a worktree), or None outside git."""
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode:
        return None
    top = Path(r.stdout.strip())
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
    if common.returncode == 0 and common.stdout.strip():  # a worktree: use the main checkout
        return Path(common.stdout.strip()).parent
    return top


# The bot's config, or None when not set up.
def config(home: Path) -> dict | None:
    """The bot's config, or None when not set up."""
    return A.load_config(home)


# The bot's config, or exit with a hint to run `bot.py setup`.
def need_config(home: Path) -> dict:
    """The bot's config, or exit with a hint to run `bot.py setup`."""
    cfg = config(home)
    if not cfg:
        sys.exit("Sanctuary Bot is not set up yet: run `bot.py setup` (from inside the repository).")
    return cfg


# The automation registry for the configured definitions folder.
def registry(home: Path) -> A.Registry:
    """The automation registry for the configured definitions folder."""
    return A.Registry(home, Path(need_config(home)["definitions_dir"]))


# Format an epoch timestamp as local `YYYY-MM-DD HH.
def when(ts: float | None) -> str:
    """Format an epoch timestamp as local `YYYY-MM-DD HH:MM`, or '-' when empty."""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


# The running daemon's pid, or None.
def daemon_pid(home: Path) -> int | None:
    """The running daemon's pid, or None."""
    pid = A.read_json(home / "state.json", {}).get("pid")
    return pid if A._alive(pid) else None


# ---------- commands ----------

# Configure the definitions folder (default.
def cmd_setup(home: Path, a: argparse.Namespace) -> int:
    """Configure the definitions folder (default: repo .agent/automations) and notify command, then run doctor."""
    root = repo_root()
    defs = Path(a.definitions).resolve() if a.definitions else (root / ".agent" / "automations" if root else None)
    if defs is None:
        print("not inside a git repository: pass --definitions DIR", file=sys.stderr)
        return 2
    defs.mkdir(parents=True, exist_ok=True)
    cfg = config(home) or {}
    cfg.update({"definitions_dir": str(defs), "configured_at": time.time()})
    if a.notify_cmd is not None:
        cfg["notify_cmd"] = shlex.split(a.notify_cmd) if a.notify_cmd else None
    home.mkdir(parents=True, exist_ok=True)
    A.write_json(home / "config.json", cfg)
    print(f"Sanctuary Bot configured. Automations folder: {defs}\nState: {home}")
    return cmd_doctor(home, a)


# Print host, config, bot-process and engine checks.
def cmd_doctor(home: Path, _a: argparse.Namespace) -> int:
    """Print host, config, bot-process and engine checks; non-zero if host or config is missing."""
    r = subprocess.run([sys.executable, str(HERE / "verify_health.py")], capture_output=True, text=True)
    checks = [("host (python, disk)", r.returncode == 0, r.stdout.strip().splitlines()[-2] if r.stdout.strip() else "")]
    cfg = config(home)
    checks.append(("configured", bool(cfg), cfg["definitions_dir"] if cfg else "run bot.py setup"))
    pid = daemon_pid(home)
    checks.append(("bot running", bool(pid), f"pid {pid}" if pid else "not running (bot.py start)"))
    checks.append(("cli-agent run_agent.py", A.Path(os.environ.get("SANCTUARY_RUN_AGENT", Path.home() / ".claude/skills/cli-agent/scripts/run_agent.py")).exists(),
                   "needed only for AI engines"))
    for eng in ("claude", "codex", "agy"):
        checks.append((f"engine {eng}", bool(shutil.which(eng)), shutil.which(eng) or "not on PATH (optional)"))
    for name, ok, detail in checks:
        print(f"[{'ok' if ok else '--'}] {name}: {detail}")
    return 0 if checks[0][1] and checks[1][1] else 1


# Start the daemon detached if it is not running.
def cmd_start(home: Path, _a: argparse.Namespace) -> int:
    """Start the daemon detached if it is not running."""
    pid = daemon_pid(home)
    if pid:
        print(f"Sanctuary Bot already running (pid {pid})")
        return 0
    home.mkdir(parents=True, exist_ok=True)
    log = open(home / "daemon.out", "ab")
    p = subprocess.Popen([sys.executable, str(DAEMON), "--home", str(home), "run"], stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if daemon_pid(home):
            break
        time.sleep(0.1)
    print(f"Sanctuary Bot started (pid {p.pid}). Stop it with: bot.py stop")
    if not config(home):
        print("Note: not set up yet, so no automations will run until `bot.py setup`.")
    return 0


# Stop the daemon (SIGTERM).
def cmd_stop(home: Path, _a: argparse.Namespace) -> int:
    """Stop the daemon (SIGTERM); a running job is killed and re-queued once."""
    pid = daemon_pid(home)
    if not pid:
        print("Sanctuary Bot is not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not A._alive(pid):
            break
        time.sleep(0.1)
    print("Sanctuary Bot stopped (a running job was killed and will re-run once on next start)")
    return 0


# One status row per automation.
def rows(home: Path) -> list[dict]:
    """One status row per automation: state, schedule, last run, next due time."""
    reg = registry(home)
    st = reg.state()
    out = []
    for n in reg.names():
        s = st.get(n, {})
        try:
            sched = reg.load(n).get("schedule", {"manual": True})
            sdesc = "manual" if sched.get("manual") else f"every {sched['every']}" + (f" at {sched['at']}" if sched.get("at") else "")
        except Exception:  # unreadable definition
            sdesc = "?"
        appr = reg.approval_status(n)
        state = ("RUNNING" if s.get("current_run") else "PAUSED" if s.get("paused") else
                 appr if appr != "APPROVED" else "ACTIVE")
        out.append({"name": n, "state": state, "schedule": sdesc, "last": f"{s.get('last_status') or '-'} {when(s.get('last_run_at'))}",
                    "next": when(s.get("next_due")) if state == "ACTIVE" else "-", "note": s.get("pause_reason") or ""})
    return out


# Print the automations table.
def cmd_list(home: Path, _a: argparse.Namespace) -> int:
    """Print the automations table."""
    rs = rows(home)
    if not rs:
        print("No automations yet. Ask your agent to set one up (it will interview you), or add a definition to "
              + need_config(home)["definitions_dir"])
        return 0
    print(f"{'NAME':22} {'STATE':24} {'SCHEDULE':18} {'LAST RUN':28} NEXT")
    for r in rs:
        print(f"{r['name']:22} {r['state']:24} {r['schedule']:18} {r['last']:28} {r['next']} {r['note']}")
    return 0


# Print bot state, the automations table and the latest inbox lines.
def cmd_status(home: Path, a: argparse.Namespace) -> int:
    """Print bot state, the automations table and the latest inbox lines."""
    pid = daemon_pid(home)
    st = A.read_json(home / "state.json", {})
    print(f"Sanctuary Bot: {'running (pid %s)' % pid if pid else 'not running'}"
          + (f", circuit {st.get('daemon_status')}" if st else ""))
    if config(home):
        cmd_list(home, a)
    inbox = home / "inbox.md"
    if inbox.exists():
        print("\nRecent results:")
        print("".join(inbox.read_text().splitlines(True)[-5:]).rstrip())
    return 0


# Print one automation's definition, approval and state as JSON.
def cmd_show(home: Path, a: argparse.Namespace) -> int:
    """Print one automation's definition, approval and state as JSON."""
    reg = registry(home)
    print(json.dumps({"definition": reg.load(a.name), "approval": reg.approvals().get(a.name),
                      "approval_status": reg.approval_status(a.name), "state": reg.state().get(a.name, {})}, indent=2))
    return 0


# Validate one definition and print its problems.
def cmd_validate(home: Path, a: argparse.Namespace) -> int:
    """Validate one definition and print its problems."""
    errs = registry(home).validate(a.name)
    print("valid" if not errs else "invalid:\n  " + "\n  ".join(errs))
    return 0 if not errs else 2


# Approve a definition for the named approver (content hash).
def cmd_approve(home: Path, a: argparse.Namespace) -> int:
    """Approve a definition for the named approver (content hash)."""
    if not a.by.strip():
        print("--by is required (the human approving this automation)", file=sys.stderr)
        return 2
    try:
        rec = registry(home).approve(a.name, a.by)
    except ValueError as e:
        print(f"not approved: {e}", file=sys.stderr)
        return 2
    print(f"{a.name} approved by {a.by} (sha256 {rec['sha256'][:12]}). Any edit to the definition pauses it until re-approved.")
    if not daemon_pid(home):
        print("The bot is not running, so it will not fire yet: bot.py start")
    return 0


# pause, resume, run-now or forget one automation.
def cmd_simple(home: Path, a: argparse.Namespace) -> int:
    """pause, resume, run-now or forget one automation."""
    reg = registry(home)
    if a.name not in reg.names():
        print(f"no automation named {a.name}", file=sys.stderr)
        return 2
    if a.cmd == "pause":
        reg.set(a.name, paused=True, pause_reason="paused by user")
    elif a.cmd == "resume":
        defn = reg.load(a.name)
        reg.set(a.name, paused=False, pause_reason=None, failed_runs=0,
                next_due=A.next_due(defn.get("schedule", {"manual": True}), time.time()))
    elif a.cmd == "run-now":
        if reg.approval_status(a.name) != "APPROVED":
            print(f"{a.name} is {reg.approval_status(a.name)}: approve it first", file=sys.stderr)
            return 2
        reg.set(a.name, run_now=True)
        if not daemon_pid(home):
            print("queued, but the bot is not running: bot.py start")
    elif a.cmd == "forget":
        ap = reg.approvals()
        ap.pop(a.name, None)
        A.write_json(reg.approvals_p, ap)
        st = reg.state()
        st.pop(a.name, None)
        A.write_json(reg.state_p, st)
        print("approval and run state removed; delete or keep the definition file through git")
        return 0
    print(f"{a.name}: {a.cmd} ok")
    return 0


# Print an automation's finished runs.
def cmd_history(home: Path, a: argparse.Namespace) -> int:
    """Print an automation's finished runs."""
    for h in registry(home).history(a.name, a.limit):
        print(f"{when(h['ended_at'])}  {h['status']:16} {h['summary']}")
    return 0


# One-off run of a definition file (job or multi-step) through the conductor, no schedule.
def cmd_run(home: Path, a: argparse.Namespace) -> int:
    """One-off run of a definition file (job or multi-step) through the conductor, no schedule."""
    defn = json.loads(Path(a.file).read_text())
    if not a.by:
        print("--by is required (who approves this one-off run)", file=sys.stderr)
        return 2
    cfg = config(home)
    if cfg and Path(a.file).resolve().parent == Path(cfg["definitions_dir"]).resolve():
        # A registered automation: run it through the scheduler so it is recorded in history and the inbox.
        reg = registry(home)
        name = Path(a.file).stem
        if reg.approval_status(name) != "APPROVED":
            print(f"{name} is {reg.approval_status(name)}: approve it first (bot.py approve {name} --by <approver>)", file=sys.stderr)
            return 2
        reg.set(name, run_now=True)
        if not daemon_pid(home):
            cmd_start(home, a)
        print(f"{name}: queued to run now; follow it with `bot.py status` and `bot.py history {name}`")
        return 0
    root = repo_root() or Path.cwd()
    name = defn.get("name", Path(a.file).stem)
    camp = A.to_campaign(name, defn, f"{name}-oneoff-{time.strftime('%Y%m%d-%H%M%S')}", a.by, root, None)
    errs = C.validate(camp)
    if errs:
        print("invalid:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    tmp = home / "campaigns" / camp["id"] / "request.json"
    A.write_json(tmp, camp)
    return C.cmd_launch(home, tmp, start_daemon=True)


# Close the daemon's global circuit breaker.
def cmd_reset(home: Path, _a: argparse.Namespace) -> int:
    """Close the daemon's global circuit breaker."""
    return subprocess.call([sys.executable, str(DAEMON), "--home", str(home), "reset"])


# Parse arguments and dispatch to the command handlers.
def main() -> int:
    """Parse arguments and dispatch to the command handlers."""
    ap = argparse.ArgumentParser(description="Sanctuary Bot: scheduled and on-demand background automations")
    ap.add_argument("--home", type=Path, default=HOME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--definitions"); s.add_argument("--notify-cmd")
    for c in ("doctor", "start", "stop", "status", "list", "reset"):
        sub.add_parser(c)
    for c in ("show", "validate", "pause", "resume", "run-now", "forget"):
        sub.add_parser(c).add_argument("name")
    h = sub.add_parser("history"); h.add_argument("name"); h.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("approve"); p.add_argument("name"); p.add_argument("--by", required=True)
    r = sub.add_parser("run"); r.add_argument("file"); r.add_argument("--by", required=True)
    a = ap.parse_args()
    table = {"setup": cmd_setup, "doctor": cmd_doctor, "start": cmd_start, "stop": cmd_stop, "status": cmd_status,
             "list": cmd_list, "show": cmd_show, "validate": cmd_validate, "approve": cmd_approve,
             "history": cmd_history, "run": cmd_run, "reset": cmd_reset}
    fn = table.get(a.cmd, cmd_simple)
    return fn(a.home, a)


if __name__ == "__main__":
    sys.exit(main())
