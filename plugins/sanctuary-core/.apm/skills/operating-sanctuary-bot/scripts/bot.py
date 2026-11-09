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


def repo_root() -> Path | None:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode:
        return None
    top = Path(r.stdout.strip())
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
    if common.returncode == 0 and common.stdout.strip():  # a worktree: use the main checkout
        return Path(common.stdout.strip()).parent
    return top


def config(home: Path) -> dict | None:
    return A.load_config(home)


def need_config(home: Path) -> dict:
    cfg = config(home)
    if not cfg:
        sys.exit("Sanctuary Bot is not set up yet: run `bot.py setup` (from inside the repository).")
    return cfg


def registry(home: Path) -> A.Registry:
    return A.Registry(home, Path(need_config(home)["definitions_dir"]))


def when(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def daemon_pid(home: Path) -> int | None:
    pid = A.read_json(home / "state.json", {}).get("pid")
    return pid if A._alive(pid) else None


# ---------- commands ----------

def cmd_setup(home: Path, a) -> int:
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


def cmd_doctor(home: Path, _a) -> int:
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


def cmd_start(home: Path, _a) -> int:
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


def cmd_stop(home: Path, _a) -> int:
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


def rows(home: Path) -> list[dict]:
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


def cmd_list(home: Path, _a) -> int:
    rs = rows(home)
    if not rs:
        print("No automations yet. Ask your agent to set one up (it will interview you), or add a definition to "
              + need_config(home)["definitions_dir"])
        return 0
    print(f"{'NAME':22} {'STATE':24} {'SCHEDULE':18} {'LAST RUN':28} NEXT")
    for r in rs:
        print(f"{r['name']:22} {r['state']:24} {r['schedule']:18} {r['last']:28} {r['next']} {r['note']}")
    return 0


def cmd_status(home: Path, a) -> int:
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


def cmd_show(home: Path, a) -> int:
    reg = registry(home)
    print(json.dumps({"definition": reg.load(a.name), "approval": reg.approvals().get(a.name),
                      "approval_status": reg.approval_status(a.name), "state": reg.state().get(a.name, {})}, indent=2))
    return 0


def cmd_validate(home: Path, a) -> int:
    errs = registry(home).validate(a.name)
    print("valid" if not errs else "invalid:\n  " + "\n  ".join(errs))
    return 0 if not errs else 2


def cmd_approve(home: Path, a) -> int:
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


def cmd_simple(home: Path, a) -> int:
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


def cmd_history(home: Path, a) -> int:
    for h in registry(home).history(a.name, a.limit):
        print(f"{when(h['ended_at'])}  {h['status']:16} {h['summary']}")
    return 0


def cmd_run(home: Path, a) -> int:
    """One-off run of a definition file (job or multi-step) through the conductor, no schedule."""
    defn = json.loads(Path(a.file).read_text())
    if not a.by:
        print("--by is required (who approves this one-off run)", file=sys.stderr)
        return 2
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


def cmd_reset(home: Path, _a) -> int:
    return subprocess.call([sys.executable, str(DAEMON), "--home", str(home), "reset"])


def main() -> int:
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
