#!/usr/bin/env python3
"""sanctuary_daemon.py - Sanctuary Bot runtime: zero-token supervisor daemon (internal; users use bot.py).

The daemon is the deterministic outer loop. It never calls a model itself. It waits on
OS file events (macOS kqueue; polling fallback elsewhere) for job files in a queue
directory, dispatches each authorized job as an isolated subprocess, and enforces rails:

  - hard wall-clock timeout per job: SIGTERM to the job's process group, 2 s grace, SIGKILL
  - circuit breaker: N consecutive failures put the daemon in BLOCKED_ON_HUMAN until `reset`
  - authorization gate: a job without `authorized_by` is REJECTED, never run
  - policy gate: no copilot backend (personal-project rule), no permission-bypass flags;
    cli-agent runs are --isolated unless the job explicitly sets allow_permission_bypass
  - crash recovery: a job left RUNNING by a stopped or crashed daemon is killed and re-queued
  - atomic state: every job/state write is temp file + os.replace

Model work happens only inside dispatched jobs. The `cli-agent` runner delegates to the
companion repo's run_agent.py, which enforces --cwd/--allowed-root, duplicate-launch
claims, its own timeout and a structured --result-json.

Daemon job states (these live in the daemon's own files, never in mission manifests):
  QUEUED -> RUNNING -> COMPLETED | FAILED | TIMED_OUT ; REJECTED for gated jobs;
  SKIPPED when a job's `after` prerequisite ended without COMPLETED.
Daemon states: ACTIVE | BLOCKED_ON_HUMAN.

Usage:
  sanctuary_daemon.py run [--home DIR] [--once]
  sanctuary_daemon.py enqueue --mission ID --authorized-by NAME (--shell CMD | --cli NAME --instruction TEXT --cwd DIR)
                             [--after JOB_ID] [--allow-permission-bypass] [--require-commit] [--timeout SEC]
  sanctuary_daemon.py status [--home DIR]
  sanctuary_daemon.py reset  [--home DIR]

Key Input Dependencies:
- ~/.sanctuary/daemon/queue/*.json job files; state.json; daemon.lock
- cli-agent run_agent.py for AI jobs (Env: SANCTUARY_RUN_AGENT; default ~/.claude/skills/cli-agent/scripts/run_agent.py)
- automations.py + ~/.sanctuary/daemon/config.json for the scheduler (optional)
- Env: SANCTUARY_DAEMON_HOME

Key Functions:
- now(): Current epoch time (seconds).
- atomic_write_json(): Write JSON atomically (fsync, then os.replace).
- read_json(): Read a JSON file, or None when missing or invalid.
- class Home: The daemon's state folder: queue, logs, claims, state.json, events.jsonl, lock.
    - __init__(): Bind to a root folder and create queue, logs and claims.
    - state(): Daemon state (ACTIVE or BLOCKED_ON_HUMAN, consecutive failures).
    - save_state(): Persist daemon state atomically.
    - event(): Append one event line to events.jsonl.
    - jobs(): All parseable job files, oldest first.
- gate(): Return a rejection reason, or None if the job may run.
- build_argv(): Build the job's command line: shell argv, or run_agent.py with the cli-agent launch controls.
- git_head(): HEAD of a git checkout, or '' outside git.
- kill_group(): SIGTERM the process group, wait GRACE_SEC, then SIGKILL.
- group_alive(): True if any live member remains. EPERM on macOS means only a zombie remains.
- class Watcher: Blocks until the queue directory changes or `timeout` elapses (kqueue, else polling).
    - __init__(): Open a kqueue watch on the queue folder when available.
    - wait(): Block until the queue changes or `timeout` seconds pass.
- class Daemon: The supervisor loop: gates and dispatches jobs, runs the scheduler while idle.
    - __init__(): Bind to a home and prepare the stop flag and warning set.
    - _on_signal(): SIGTERM/SIGINT handler: request a clean stop.
    - recover(): Kill and re-queue jobs left RUNNING by a stopped or crashed daemon.
    - run(): Take the single-instance lock, recover, then loop: dispatch jobs or idle on the watcher.
    - _scheduler(): Sanctuary Bot automations, if `bot.py setup` configured a definitions directory.
    - tick(): Gate and dispatch the oldest runnable QUEUED job; returns True if one ran.
    - dispatch(): Launch a job, wait for it under the rails, and record its final state.
    - _wait(): Wait for the job; returns (status or None for a normal exit, exit code).
    - _requeue_interrupted(): Record an interrupted job so the next daemon start re-runs it once.
    - _count_outcome(): Update the consecutive-failure counter and open the circuit at the limit.
- cmd_enqueue(): Write one QUEUED job file from command-line arguments; prints its id.
- main(): Parse arguments and dispatch run, enqueue, status or reset.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEFAULT_HOME = Path(os.environ.get("SANCTUARY_DAEMON_HOME", Path.home() / ".sanctuary" / "daemon"))
DEFAULT_TIMEOUT_SEC = 600
GRACE_SEC = 2.0
MAX_CONSECUTIVE_FAILURES = 3
MAX_ATTEMPTS = 2
POLL_FALLBACK_SEC = 0.5
DENIED_CLIS = {"copilot"}  # user rule: Copilot is for work repositories only
FORBIDDEN_TOKENS = ("--dangerously-skip-permissions", "--yolo",
                    "--dangerously-bypass-approvals-and-sandbox", "--allow-copilot")
RUN_AGENT = Path(os.environ.get("SANCTUARY_RUN_AGENT",
                                Path.home() / ".claude/skills/cli-agent/scripts/run_agent.py"))


# ---------- small helpers ----------

# Current epoch time (seconds).
def now() -> float:
    """Current epoch time (seconds)."""
    return time.time()


# Write JSON atomically (fsync, then os.replace).
def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (fsync, then os.replace)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# Read a JSON file, or None when missing or invalid.
def read_json(path: Path) -> dict | None:
    """Read a JSON file, or None when missing or invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# The daemon's state folder.
class Home:
    """The daemon's state folder: queue, logs, claims, state.json, events.jsonl, lock."""
    # Bind to a root folder and create queue, logs and claims.
    def __init__(self, root: Path) -> None:
        """Bind to a root folder and create queue, logs and claims."""
        self.root = root
        self.queue = root / "queue"
        self.logs = root / "logs"
        self.claims = root / "claims"
        self.state_file = root / "state.json"
        self.events = root / "events.jsonl"
        self.lock_file = root / "daemon.lock"
        for d in (self.queue, self.logs, self.claims):
            d.mkdir(parents=True, exist_ok=True)

    # Daemon state (ACTIVE or BLOCKED_ON_HUMAN, consecutive failures).
    def state(self) -> dict:
        """Daemon state (ACTIVE or BLOCKED_ON_HUMAN, consecutive failures)."""
        return read_json(self.state_file) or {"daemon_status": "ACTIVE", "consecutive_failures": 0}

    # Persist daemon state atomically.
    def save_state(self, s: dict) -> None:
        """Persist daemon state atomically."""
        atomic_write_json(self.state_file, s)

    # Append one event line to events.jsonl.
    def event(self, kind: str, **fields) -> None:
        """Append one event line to events.jsonl."""
        with open(self.events, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now(), "event": kind, **fields}) + "\n")

    # All parseable job files, oldest first.
    def jobs(self) -> list[tuple[Path, dict]]:
        """All parseable job files, oldest first."""
        out = []
        for p in self.queue.glob("*.json"):
            j = read_json(p)
            if j is not None:
                out.append((p, j))
        return sorted(out, key=lambda pj: (pj[1].get("created_at", 0), pj[1].get("id", "")))


# ---------- gates ----------

# Return a rejection reason, or None if the job may run.
def gate(job: dict) -> str | None:
    """Return a rejection reason, or None if the job may run."""
    if not str(job.get("authorized_by", "")).strip():
        return "missing authorized_by (Human Gate)"
    runner = job.get("runner") or {}
    rtype = runner.get("type")
    if rtype not in ("shell", "cli-agent"):
        return f"unknown runner type {rtype!r}"
    text = json.dumps(runner)
    for tok in FORBIDDEN_TOKENS:
        if tok in text:
            return f"forbidden flag {tok}"
    if rtype == "cli-agent":
        if runner.get("cli") in DENIED_CLIS:
            return f"backend {runner.get('cli')!r} is not allowed for this repository"
        for key in ("cli", "instruction", "cwd"):
            if not runner.get(key):
                return f"cli-agent runner missing {key}"
        # run_agent.py adds --dangerously-skip-permissions for claude/agy unless --isolated.
        if runner.get("isolated") is False and job.get("allow_permission_bypass") is not True:
            return "non-isolated cli-agent run needs explicit allow_permission_bypass (run_agent adds a permission bypass)"
    else:
        if not runner.get("argv"):
            return "shell runner missing argv"
    return None


# Build the job's command line.
def build_argv(home: Home, job: dict) -> tuple[list[str], str]:
    """Build the job's command line: shell argv, or run_agent.py with the cli-agent launch controls."""
    runner = job["runner"]
    if runner["type"] == "shell":
        return list(runner["argv"]), runner.get("cwd") or str(home.root)
    cwd = str(Path(runner["cwd"]).resolve())
    out_md = home.logs / f"{job['id']}.out.md"
    result = home.logs / f"{job['id']}.result.json"
    argv = [sys.executable, str(RUN_AGENT), "/dev/null", "/dev/null", str(out_md), runner["instruction"],
            "--cli", runner["cli"]]
    if not runner.get("clean_context"):  # clean context picks its own neutral cwd; run_agent refuses --cwd with it
        argv += ["--cwd", cwd, "--allowed-root", str(Path(runner.get("allowed_root") or cwd).resolve())]
    argv += ["--claim-dir", str(home.claims), "--unit", job["id"], "--owner", "sanctuary-daemon",
            "--timeout", str(int(job.get("timeout_sec", DEFAULT_TIMEOUT_SEC))),
            "--result-json", str(result)]
    if runner.get("model"):
        argv += ["--model", runner["model"]]
    if runner.get("isolated", True):
        argv.append("--isolated")
    if runner.get("clean_context"):
        argv.append("--clean-context")  # claude: neutral cwd, no tools: for untrusted inputs  # default: no permission-bypass flags from run_agent
    for flag in ("effort", "mode", "tools"):
        if runner.get(flag):
            argv += [f"--{flag}", str(runner[flag])]
    return argv, cwd


# HEAD of a git checkout, or '' outside git.
def git_head(cwd: str) -> str:
    """HEAD of a git checkout, or '' outside git."""
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


# SIGTERM the process group, wait GRACE_SEC, then SIGKILL.
def kill_group(pgid: int, proc: subprocess.Popen | None = None) -> None:
    """SIGTERM the process group, wait GRACE_SEC, then SIGKILL.

    `proc` (the group leader, our child) is reaped while waiting: on macOS, signalling a
    group whose only remaining member is an unreaped zombie leader raises EPERM.
    """
    for sig, wait in ((signal.SIGTERM, GRACE_SEC), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            if proc is not None:
                proc.poll()
            if not group_alive(pgid):
                return
        deadline = now() + wait
        while now() < deadline:
            if proc is not None:
                proc.poll()
            if not group_alive(pgid):
                return
            time.sleep(0.05)


# True if any live member remains.
def group_alive(pgid: int) -> bool:
    """True if any live member remains. EPERM on macOS means only a zombie remains."""
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---------- watcher ----------

# Blocks until the queue directory changes or `timeout` elapses (kqueue, else polling).
class Watcher:
    """Blocks until the queue directory changes or `timeout` elapses (kqueue, else polling)."""

    # Open a kqueue watch on the queue folder when available.
    def __init__(self, path: Path) -> None:
        """Open a kqueue watch on the queue folder when available."""
        self.path = path
        self.kq = None
        if hasattr(select, "kqueue"):
            self.fd = os.open(path, os.O_RDONLY)
            self.kq = select.kqueue()
            self.kev = select.kevent(self.fd, filter=select.KQ_FILTER_VNODE,
                                     flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                     fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND)
            self.kind = "kqueue"
        else:
            self.kind = "poll"

    # Block until the queue changes or `timeout` seconds pass.
    def wait(self, timeout: float) -> None:
        """Block until the queue changes or `timeout` seconds pass."""
        if self.kq is not None:
            self.kq.control([self.kev], 1, timeout)
        else:
            time.sleep(min(timeout, POLL_FALLBACK_SEC))


# ---------- daemon ----------

# The supervisor loop.
class Daemon:
    """The supervisor loop: gates and dispatches jobs, runs the scheduler while idle."""
    # Bind to a home and prepare the stop flag and warning set.
    def __init__(self, home: Home) -> None:
        """Bind to a home and prepare the stop flag and warning set."""
        self.home = home
        self.stop = False
        self.current: tuple[Path, dict, subprocess.Popen] | None = None
        self.warned: set[str] = set()

    # SIGTERM/SIGINT handler.
    def _on_signal(self, signum: int, _frame: object) -> None:
        """SIGTERM/SIGINT handler: request a clean stop."""
        self.stop = True

    # Kill and re-queue jobs left RUNNING by a stopped or crashed daemon.
    def recover(self) -> None:
        """Kill and re-queue jobs left RUNNING by a stopped or crashed daemon."""
        for path, job in self.home.jobs():
            if job.get("status") == "RUNNING":
                pgid = job.get("pgid")
                if pgid and group_alive(pgid):
                    kill_group(pgid)
                attempts = job.get("attempts", 1)
                job["status"] = "QUEUED" if attempts < job.get("max_attempts", MAX_ATTEMPTS) else "FAILED"
                job["recovered_at"] = now()
                job.setdefault("history", []).append({"ts": now(), "note": "recovered after daemon stop"})
                atomic_write_json(path, job)
                self.home.event("recovered", job=job["id"], status=job["status"])

    # Take the single-instance lock, recover, then loop.
    def run(self, once: bool = False) -> int:
        """Take the single-instance lock, recover, then loop: dispatch jobs or idle on the watcher."""
        lock = open(self.home.lock_file, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another sanctuary daemon holds the lock", file=sys.stderr)
            return 2
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        st = self.home.state()
        st.update({"pid": os.getpid(), "started_at": now()})
        self.home.save_state(st)
        self.recover()
        watcher = Watcher(self.home.queue)
        scheduler = None if once else self._scheduler()
        self.home.event("daemon_start", pid=os.getpid(), watcher=watcher.kind, scheduler=bool(scheduler))
        while not self.stop:
            ran = self.tick()
            if once and not ran:
                break
            if not ran and scheduler is not None:
                try:
                    for msg in scheduler.tick():
                        self.home.event("automation", msg=msg)
                except Exception as exc:  # a bad definition must not stop the daemon
                    self.home.event("scheduler_error", error=str(exc)[:300])
            if not ran and not self.stop:
                watcher.wait(1.0)  # zero tokens, ~zero CPU while idle
        self.home.event("daemon_stop", pid=os.getpid())
        return 0

    # Sanctuary Bot automations, if `bot.py setup` configured a definitions directory.
    def _scheduler(self) -> object | None:
        """Sanctuary Bot automations, if `bot.py setup` configured a definitions directory."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import automations
        except ImportError:
            return None
        cfg = automations.load_config(self.home.root)
        return automations.Scheduler(self.home.root, cfg) if cfg else None

    # Gate and dispatch the oldest runnable QUEUED job.
    def tick(self) -> bool:
        """Gate and dispatch the oldest runnable QUEUED job; returns True if one ran."""
        st = self.home.state()
        jobs = self.home.jobs()
        by_id = {j.get("id"): j for _, j in jobs}
        for path, job in jobs:
            if job.get("status") != "QUEUED":
                continue
            dep = job.get("after")
            if dep:
                prereq = by_id.get(dep)
                pstat = prereq.get("status") if prereq else None
                if pstat in (None, "QUEUED", "RUNNING"):
                    if prereq is None and job.get("id") not in self.warned:
                        self.warned.add(job.get("id"))
                        self.home.event("waiting_missing_prereq", job=job.get("id"), after=dep)
                    continue  # wait for the prerequisite
                if pstat != "COMPLETED":
                    job.update({"status": "SKIPPED", "reason": f"prerequisite {dep} ended {pstat}", "ended_at": now()})
                    atomic_write_json(path, job)
                    self.home.event("skipped", job=job.get("id"), after=dep, prereq_status=pstat)
                    continue
            reason = gate(job)
            if reason:
                job.update({"status": "REJECTED", "reason": reason, "ended_at": now()})
                atomic_write_json(path, job)
                self.home.event("rejected", job=job.get("id"), reason=reason)
                continue
            if st.get("daemon_status") == "BLOCKED_ON_HUMAN":
                return False  # circuit open: leave QUEUED, do not dispatch
            self.dispatch(path, job)
            return True
        return False

    # Run one gated job in its own process group, enforce its timeout, record the outcome.
    def dispatch(self, path: Path, job: dict) -> None:
        """Launch a job, wait for it under the rails, and record its final state."""
        argv, cwd = build_argv(self.home, job)
        log = open(self.home.logs / f"{job['id']}.log", "ab")
        head_before = git_head(cwd) if job.get("require_commit") else ""
        proc = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        job.update({"status": "RUNNING", "running_at": now(), "pid": proc.pid, "pgid": proc.pid,
                    "attempts": job.get("attempts", 0) + 1})
        atomic_write_json(path, job)
        self.home.event("dispatch", job=job["id"], mission=job.get("mission_id"), pid=proc.pid)
        status, code = self._wait(proc, job)
        log.close()
        if status == "INTERRUPTED":
            self._requeue_interrupted(path, job)
            return
        if status is None:
            committed = (git_head(cwd) != head_before) if job.get("require_commit") else True
            status = "COMPLETED" if code == 0 and committed else "FAILED"
            if code == 0 and not committed:
                job["reason"] = "exit 0 but no new commit (require_commit)"
        job.update({"status": status, "exit_code": code, "ended_at": now(), "pgid_alive_after": group_alive(proc.pid)})
        atomic_write_json(path, job)
        self._count_outcome(status)
        self.home.event("finished", job=job["id"], status=status, exit_code=code)

    # Poll until exit, daemon stop (INTERRUPTED) or timeout (TIMED_OUT, process group killed).
    def _wait(self, proc: subprocess.Popen, job: dict) -> tuple[str | None, int | None]:
        """Wait for the job; returns (status or None for a normal exit, exit code)."""
        timeout = float(job.get("timeout_sec", DEFAULT_TIMEOUT_SEC))
        while True:
            code = proc.poll()
            if code is not None:
                return None, code
            if self.stop:  # daemon shutdown: kill job, recover on next start
                kill_group(proc.pid, proc)
                proc.wait()
                return "INTERRUPTED", proc.returncode
            if now() - job["running_at"] > timeout:
                kill_group(proc.pid, proc)
                proc.wait()
                return "TIMED_OUT", proc.returncode
            time.sleep(0.1)

    # A job killed by daemon shutdown goes back to QUEUED (or FAILED after max attempts).
    def _requeue_interrupted(self, path: Path, job: dict) -> None:
        """Record an interrupted job so the next daemon start re-runs it once."""
        job.update({"status": "QUEUED" if job["attempts"] < job.get("max_attempts", MAX_ATTEMPTS) else "FAILED",
                    "interrupted_at": now()})
        job.setdefault("history", []).append({"ts": now(), "note": "interrupted by daemon stop"})
        atomic_write_json(path, job)
        self.home.event("interrupted", job=job["id"])

    # Global breaker: consecutive non-COMPLETED jobs open the circuit (BLOCKED_ON_HUMAN).
    def _count_outcome(self, status: str) -> None:
        """Update the consecutive-failure counter and open the circuit at the limit."""
        st = self.home.state()
        if status == "COMPLETED":
            st["consecutive_failures"] = 0
        else:
            st["consecutive_failures"] = st.get("consecutive_failures", 0) + 1
            if st["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                st["daemon_status"] = "BLOCKED_ON_HUMAN"
                self.home.event("circuit_open", failures=st["consecutive_failures"])
        self.home.save_state(st)


# ---------- CLI ----------

# Write one QUEUED job file from command-line arguments.
def cmd_enqueue(home: Home, a: argparse.Namespace) -> int:
    """Write one QUEUED job file from command-line arguments; prints its id."""
    jid = a.id or f"{a.mission}-{uuid.uuid4().hex[:8]}"
    if a.shell:
        runner = {"type": "shell", "argv": shlex.split(a.shell), "cwd": a.cwd}
    else:
        runner = {"type": "cli-agent", "cli": a.cli, "model": a.model, "instruction": a.instruction,
                  "cwd": a.cwd, "allowed_root": a.allowed_root, "effort": a.effort, "mode": a.mode,
                  "clean_context": a.clean_context}
    if a.allow_permission_bypass:
        if a.shell:
            print("--allow-permission-bypass applies only to cli-agent jobs", file=sys.stderr)
            return 2
        runner["isolated"] = False
    job = {"id": jid, "mission_id": a.mission, "status": "QUEUED", "authorized_by": a.authorized_by,
           "created_at": now(), "timeout_sec": a.timeout, "require_commit": a.require_commit,
           "max_attempts": MAX_ATTEMPTS, "runner": runner}
    if a.after:
        job["after"] = a.after
    if a.allow_permission_bypass:
        job["allow_permission_bypass"] = True
    atomic_write_json(home.queue / f"{jid}.json", job)
    print(jid)
    return 0


# Parse arguments and dispatch run, enqueue, status or reset.
def main() -> int:
    """Parse arguments and dispatch run, enqueue, status or reset."""
    ap = argparse.ArgumentParser(description="Sanctuary zero-token supervisor daemon (MISSION-001 POC)")
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the daemon in the foreground")
    r.add_argument("--once", action="store_true", help="drain the queue then exit")
    e = sub.add_parser("enqueue", help="queue one job")
    e.add_argument("--mission", required=True)
    e.add_argument("--authorized-by", default="")
    e.add_argument("--id")
    e.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    e.add_argument("--require-commit", action="store_true")
    e.add_argument("--cwd")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--shell", help="command line for the shell runner")
    g.add_argument("--cli", help="cli-agent backend (claude, codex, agy, llama, ...)")
    e.add_argument("--model")
    e.add_argument("--instruction")
    e.add_argument("--allowed-root")
    e.add_argument("--effort")
    e.add_argument("--mode")
    e.add_argument("--clean-context", action="store_true", help="cli-agent claude only: no tools, neutral cwd")
    e.add_argument("--after", help="job id that must COMPLETE first; otherwise this job is SKIPPED")
    e.add_argument("--allow-permission-bypass", action="store_true",
                   help="cli-agent only: run non-isolated; run_agent then passes the engine's permission-bypass "
                        "flag (needed for unattended edits by claude/agy). Requires explicit human approval.")
    sub.add_parser("status", help="print daemon state and jobs")
    sub.add_parser("reset", help="close the circuit breaker after human review")
    a = ap.parse_args()
    home = Home(a.home)
    if a.cmd == "run":
        return Daemon(home).run(once=a.once)
    if a.cmd == "enqueue":
        return cmd_enqueue(home, a)
    if a.cmd == "status":
        print(json.dumps({"state": home.state(),
                          "jobs": [{k: j.get(k) for k in ("id", "mission_id", "status", "exit_code", "reason")}
                                   for _, j in home.jobs()]}, indent=2))
        return 0
    if a.cmd == "reset":
        st = home.state()
        st.update({"daemon_status": "ACTIVE", "consecutive_failures": 0, "reset_at": now()})
        home.save_state(st)
        home.event("reset")
        print("circuit closed")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
