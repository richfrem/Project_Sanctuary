#!/usr/bin/env python3
"""conductor.py - zero-token multi-step conductor for Sanctuary Bot runs (internal; users use bot.py).

The conductor is the "brain": it picks the next step (triage), assembles the step prompt from
the git trail and the last verification, hands the step to the daemon (the "muscle": gates,
timeouts, process-group kill, crash recovery) through its public `enqueue` CLI, then evaluates
the outcome itself: job completed, commit made, only allowed paths touched, verify command
passed. A failed step can be reverted (dedicated task worktree only) and retried once with the
failure in its prompt. The conductor never calls a model.

Modes: pipeline (declared order + depends_on), backlog (highest priority first; a step can
inherit a linked mission's priority_score), sweep (verify -> fix until verify passes).

Usage:
  conductor.py validate MANIFEST
  conductor.py launch   MANIFEST [--start-daemon] [--home DIR]   # detached; prints PIDs and logs
  conductor.py run      CAMPAIGN_ID [--home DIR]                  # foreground conductor (launch uses this)
  conductor.py status   [CAMPAIGN_ID] [--home DIR]
  conductor.py stop     CAMPAIGN_ID [--home DIR]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage  # noqa: E402

HOME = Path(os.environ.get("SANCTUARY_DAEMON_HOME", Path.home() / ".sanctuary" / "daemon"))
DAEMON = Path(os.environ.get("SANCTUARY_DAEMON_SCRIPT", Path(__file__).resolve().parent / "sanctuary_daemon.py"))
DEFAULT_RAILS = {"max_cycles": 5, "step_timeout_sec": 300, "max_consecutive_failures": 2,
                 "require_commit": True, "allowed_paths": ["**"], "verify_cmd": None,
                 "verify_timeout_sec": 300, "verify_scope": "each_step", "revert_on_failure": False,
                 "max_retries_per_step": 1}
ENGINES = {"shell", "claude", "codex", "agy", "llama"}


# ---------- helpers ----------

def atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def git(wt: str, *args, check=False) -> str:
    r = subprocess.run(["git", "-C", wt, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def daemon_running(home: Path) -> bool:
    try:
        return pid_alive(json.loads((home / "state.json").read_text()).get("pid"))
    except (OSError, json.JSONDecodeError):
        return False


# ---------- validation ----------

def validate(m: dict) -> list[str]:
    errs = []
    for k in ("id", "mode", "authorized_by", "worktree", "engine"):
        if not m.get(k):
            errs.append(f"missing {k}")
    if m.get("mode") not in ("pipeline", "backlog", "sweep"):
        errs.append("mode must be pipeline, backlog or sweep")
    eng = m.get("engine") or {}
    if eng.get("cli") not in ENGINES:
        errs.append(f"engine.cli must be one of {sorted(ENGINES)} (copilot is not allowed)")
    if eng.get("cli") != "shell" and not eng.get("model") and eng.get("cli") == "claude":
        errs.append("engine.model is required for claude")
    rails = {**DEFAULT_RAILS, **m.get("rails", {})}
    if m.get("mode") == "sweep":
        if not rails.get("verify_cmd"):
            errs.append("sweep mode needs rails.verify_cmd")
        if not (m.get("sweep") or {}).get("fix_goal") and not (m.get("sweep") or {}).get("fix_shell"):
            errs.append("sweep mode needs sweep.fix_goal (agent) or sweep.fix_shell (shell)")
    else:
        steps = m.get("steps") or []
        if not steps:
            errs.append("pipeline/backlog mode needs steps")
        ids = [s.get("id") for s in steps]
        if len(ids) != len(set(ids)) or None in ids:
            errs.append("step ids must be present and unique")
        for s in steps:
            for d in s.get("depends_on", []):
                if d not in ids:
                    errs.append(f"step {s.get('id')} depends on unknown {d}")
            if eng.get("cli") == "shell" and not s.get("shell"):
                errs.append(f"step {s.get('id')} needs `shell` for a shell engine")
            if eng.get("cli") != "shell" and not s.get("goal"):
                errs.append(f"step {s.get('id')} needs `goal`")
    wt = m.get("worktree")
    if wt and not Path(wt).is_dir():
        errs.append(f"worktree {wt} does not exist")
    if rails["revert_on_failure"] and wt and not revert_allowed(wt):
        errs.append("revert_on_failure needs a dedicated task worktree (.worktrees/<task>, not main/master)")
    if rails["verify_scope"] not in ("each_step", "final"):
        errs.append("rails.verify_scope must be each_step or final")
    if int(rails["max_cycles"]) < 1 or int(rails["step_timeout_sec"]) < 1:
        errs.append("max_cycles and step_timeout_sec must be >= 1")
    return errs


def revert_allowed(wt: str) -> bool:
    p = str(Path(wt).resolve())
    branch = git(wt, "rev-parse", "--abbrev-ref", "HEAD")
    top = git(wt, "rev-parse", "--show-toplevel")
    return "/.worktrees/" in p + "/" and branch not in ("main", "master", "HEAD", "") and \
        Path(top).resolve() == Path(p)


# ---------- conductor ----------

class Conductor:
    def __init__(self, home: Path, cid: str):
        self.home, self.cid = home, cid
        self.dir = home / "campaigns" / cid
        self.m = json.loads((self.dir / "manifest.json").read_text())
        self.rails = {**DEFAULT_RAILS, **self.m.get("rails", {})}
        self.wt = str(Path(self.m["worktree"]).resolve())
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "campaign": cid, "status": "RUNNING", "cycles": 0, "consecutive_failures": 0,
            "steps": [{**s, "status": "PENDING", "attempts": 0} for s in self.m.get("steps", [])],
            "history": [], "last_summary": None}
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))

    def save(self):
        self.state["updated_at"] = time.time()
        atomic_write(self.state_path, self.state)

    def log(self, msg: str, **kw):
        with open(self.dir / "conductor.log", "a") as f:
            f.write(json.dumps({"ts": time.time(), "msg": msg, **kw}) + "\n")

    # --- verification and git trail ---
    def verify(self, cmd: str | None = None) -> tuple[bool | None, str]:
        cmd = cmd if cmd is not None else self.rails.get("verify_cmd")
        if not cmd:
            return None, ""
        try:
            r = subprocess.run(cmd, shell=True, cwd=self.wt, capture_output=True, text=True,
                               timeout=int(self.rails["verify_timeout_sec"]))
            return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            return False, "verify command timed out"

    def trail(self, pre: str, post: str) -> dict:
        if not pre or pre == post:
            return {"commit": None, "files": []}
        files = [l for l in git(self.wt, "diff", "--name-only", pre, post).splitlines() if l]
        subj = git(self.wt, "log", "-1", "--pretty=%s (%h)", post)
        return {"commit": subj, "files": files}

    def prompt(self, goal: str, retry_ctx: str | None) -> str:
        parts = []
        if self.m.get("previous_run"):
            parts.append("PREVIOUS RUN OF THIS AUTOMATION: " + self.m["previous_run"])
        ls = self.state.get("last_summary")
        if ls:
            parts.append("PREVIOUS STEP: " + ls)
        if retry_ctx:
            parts.append("RETRY: the previous attempt at this step failed and was "
                         + ("reverted" if self.rails["revert_on_failure"] else "left in place")
                         + f". Failure: {retry_ctx}. Fix that first.")
        parts.append("GOAL: " + goal)
        rules = [f"Work only inside {self.wt}."]
        if self.rails["allowed_paths"] != ["**"]:
            rules.append("Change only these paths: " + ", ".join(self.rails["allowed_paths"]) + ".")
        if self.rails.get("verify_cmd"):
            rules.append(f"Make `{self.rails['verify_cmd']}` pass.")
        if self.rails["require_commit"]:
            rules.append("Commit your work with a clear message.")
        parts.append("RULES: " + " ".join(rules))
        return "\n\n".join(parts)

    # --- dispatch through the daemon ---
    def dispatch(self, step_id: str, goal: str | None, shell_cmd: str | None, retry_ctx: str | None) -> dict:
        jid = f"{self.cid}-{step_id}-c{self.state['cycles'] + 1}"
        ctx_file = self.dir / f"{jid}.context.txt"
        text = self.prompt(goal or shell_cmd or "", retry_ctx)
        ctx_file.write_text(text)
        eng = self.m["engine"]
        cmd = [sys.executable, str(DAEMON), "--home", str(self.home), "enqueue", "--mission",
               step_mission(self.m, step_id), "--authorized-by", self.m["authorized_by"], "--id", jid,
               "--timeout", str(self.rails["step_timeout_sec"]), "--cwd", self.wt]
        if eng["cli"] == "shell":
            # Run through the platform shell so &&, |, > and quoting behave as written.
            line = shell_cmd.replace("{context_file}", str(ctx_file))
            wrapped = f"cmd /c {line}" if os.name == "nt" else "sh -c " + shlex.quote(line)
            cmd += ["--shell", wrapped]
        else:
            cmd += ["--cli", eng["cli"], "--instruction", text]
            if eng.get("model"):
                cmd += ["--model", eng["model"]]
            if eng.get("allow_permission_bypass"):
                cmd += ["--allow-permission-bypass"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return {"status": "ENQUEUE_FAILED", "reason": r.stderr.strip()[-300:]}
        job_path = self.home / "queue" / f"{jid}.json"
        while not self.stop:
            try:
                job = json.loads(job_path.read_text())
            except (OSError, json.JSONDecodeError):
                job = {}
            if job.get("status") in ("COMPLETED", "FAILED", "TIMED_OUT", "REJECTED", "SKIPPED"):
                return job
            time.sleep(0.5)  # zero tokens while waiting
        return {"status": "STOPPED"}

    def step_verify_cmd(self, step: dict | None) -> str | None:
        """Per-step `verify` wins; otherwise the campaign verify_cmd, which under
        verify_scope=final runs only when this is the last pending step."""
        if step and step.get("verify"):
            return step["verify"]
        if self.rails["verify_scope"] == "final" and step is not None:
            others = [s for s in self.state["steps"] if s is not step and s.get("status") in ("PENDING", "RUNNING")]
            if others:
                return ""
        return self.rails.get("verify_cmd")

    def evaluate(self, job: dict, pre: str, step: dict | None = None) -> tuple[bool, str, dict]:
        post = git(self.wt, "rev-parse", "HEAD")
        tr = self.trail(pre, post)
        if job.get("status") != "COMPLETED":
            return False, f"daemon job {job.get('status')}: {job.get('reason') or 'exit ' + str(job.get('exit_code'))}", tr
        if self.rails["require_commit"] and not tr["commit"]:
            return False, "no new commit", tr
        bad = [f for f in tr["files"] if not any(fnmatch.fnmatch(f, g) for g in self.rails["allowed_paths"])]
        if bad:
            return False, "touched paths outside allowed_paths: " + ", ".join(bad[:10]), tr
        ok, out = self.verify(self.step_verify_cmd(step) or None) if self.step_verify_cmd(step) != "" else (None, "")
        tr["verify"] = None if ok is None else ("passed" if ok else "failed")
        if ok is False:
            return False, "verify failed: " + out.strip().splitlines()[-1][:300] if out.strip() else "verify failed", tr
        return True, "ok", tr

    def revert(self, pre: str):
        if self.rails["revert_on_failure"] and revert_allowed(self.wt) and pre:
            git(self.wt, "reset", "--hard", pre)
            git(self.wt, "clean", "-fd")
            self.log("reverted", to=pre)

    def record(self, step_id, ok, reason, tr):
        if ok:
            self.state["last_summary"] = (f"step {step_id} committed {tr.get('commit')}; files: "
                                          f"{', '.join(tr.get('files', [])[:15]) or 'none'}; verify: {tr.get('verify') or 'n/a'}.")
        self.state["history"].append({"ts": time.time(), "step": step_id, "ok": ok, "reason": reason, **tr})

    def fail(self) -> bool:
        self.state["consecutive_failures"] += 1
        if self.state["consecutive_failures"] >= int(self.rails["max_consecutive_failures"]):
            self.state["status"] = "BLOCKED_ON_HUMAN"
            self.log("circuit open", failures=self.state["consecutive_failures"])
            return True
        return False

    def run(self) -> str:
        self.log("conductor start", pid=os.getpid(), mode=self.m["mode"])
        self.state["pid"] = os.getpid()
        self.save()
        mode = self.m["mode"]
        while self.state["status"] == "RUNNING":
            if self.stop:
                self.state["status"] = "STOPPED"
                break
            if self.state["cycles"] >= int(self.rails["max_cycles"]):
                self.state["status"] = "EXHAUSTED"
                break
            if mode == "sweep":
                ok, out = self.verify()
                if ok:
                    self.state["status"] = "COMPLETED"
                    break
                sw = self.m["sweep"]
                pre = git(self.wt, "rev-parse", "HEAD")
                ctx = "verify output tail: " + out.strip()[-600:]
                job = self.dispatch("sweep", sw.get("fix_goal"), sw.get("fix_shell"), ctx)
                self.state["cycles"] += 1
                good, reason, tr = self.evaluate(job, pre) if job.get("status") != "STOPPED" else (False, "stopped", {})
                self.record("sweep", good, reason, tr)
                if good:
                    self.state["consecutive_failures"] = 0
                else:
                    self.revert(pre)
                    self.fail()
                self.save()
                continue
            step = triage.next_step(self.state["steps"], mode, Path(git(self.wt, "rev-parse", "--show-toplevel") or self.wt))
            if step is None:
                self.state["status"] = triage.outcome(self.state["steps"])
                break
            pre = git(self.wt, "rev-parse", "HEAD")
            step["status"] = "RUNNING"
            step["attempts"] += 1
            self.save()
            job = self.dispatch(step["id"], step.get("goal"), step.get("shell"), step.get("retry_context"))
            self.state["cycles"] += 1
            if job.get("status") == "STOPPED":
                step["status"] = "PENDING"
                self.state["status"] = "STOPPED"
                break
            good, reason, tr = self.evaluate(job, pre, step)
            self.record(step["id"], good, reason, tr)
            if good:
                step["status"] = "DONE"
                step.pop("retry_context", None)
                self.state["consecutive_failures"] = 0
            else:
                self.revert(pre)
                if step["attempts"] <= int(self.rails["max_retries_per_step"]):
                    step["status"], step["retry_context"] = "PENDING", reason
                else:
                    step["status"], step["reason"] = "FAILED", reason
                self.fail()
            self.save()
        triage.propagate_blocks(self.state["steps"])  # final view: dependents of failures show BLOCKED
        self.state["ended_at"] = time.time()
        self.save()
        self.log("conductor end", status=self.state["status"])
        return self.state["status"]


def step_mission(m: dict, step_id: str) -> str:
    for s in m.get("steps", []):
        if s.get("id") == step_id and s.get("mission_id"):
            return s["mission_id"]
    return m.get("mission_id") or f"CAMPAIGN-{m['id']}"


# ---------- CLI ----------

def cmd_launch(home: Path, manifest: Path, start_daemon: bool) -> int:
    m = json.loads(manifest.read_text())
    errs = validate(m)
    if errs:
        print("invalid campaign:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    d = home / "campaigns" / m["id"]
    if (d / "state.json").exists() and json.loads((d / "state.json").read_text()).get("status") == "RUNNING" \
            and pid_alive(json.loads((d / "state.json").read_text()).get("pid")):
        print(f"campaign {m['id']} is already running", file=sys.stderr)
        return 2
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").unlink(missing_ok=True)
    atomic_write(d / "manifest.json", {**m, "worktree": str(Path(m["worktree"]).resolve())})
    out = {"campaign": m["id"], "home": str(home)}
    if not daemon_running(home):
        if not start_daemon:
            print("daemon not running: start it (sanctuary_daemon.py run) or pass --start-daemon", file=sys.stderr)
            return 3
        dlog = open(home / "daemon.out", "ab")
        dp = subprocess.Popen([sys.executable, str(DAEMON), "--home", str(home), "run"], stdout=dlog,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(50):
            if daemon_running(home):
                break
            time.sleep(0.1)
        out["daemon_pid"] = dp.pid
    clog = open(d / "conductor.out", "ab")
    cp = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--home", str(home), "run", m["id"]],
                          stdout=clog, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    out.update({"conductor_pid": cp.pid, "state": str(d / "state.json"), "log": str(d / "conductor.log"),
                "job_logs": str(home / "logs"), "stop": f"conductor.py stop {m['id']}"})
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanctuary daemon campaign conductor")
    ap.add_argument("--home", type=Path, default=HOME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("manifest", type=Path)
    la = sub.add_parser("launch"); la.add_argument("manifest", type=Path); la.add_argument("--start-daemon", action="store_true")
    r = sub.add_parser("run"); r.add_argument("campaign")
    st = sub.add_parser("status"); st.add_argument("campaign", nargs="?")
    sp = sub.add_parser("stop"); sp.add_argument("campaign")
    a = ap.parse_args()
    home = a.home
    if a.cmd == "validate":
        errs = validate(json.loads(a.manifest.read_text()))
        print("valid" if not errs else "invalid:\n  " + "\n  ".join(errs))
        return 0 if not errs else 2
    if a.cmd == "launch":
        return cmd_launch(home, a.manifest, a.start_daemon)
    if a.cmd == "run":
        status = Conductor(home, a.campaign).run()
        return 0 if status == "COMPLETED" else 1
    if a.cmd == "status":
        root = home / "campaigns"
        ids = [a.campaign] if a.campaign else sorted(p.name for p in root.glob("*")) if root.exists() else []
        out = []
        for cid in ids:
            p = root / cid / "state.json"
            s = json.loads(p.read_text()) if p.exists() else {"status": "NOT_STARTED"}
            out.append({"campaign": cid, "status": s.get("status"), "cycles": s.get("cycles"),
                        "running": pid_alive(s.get("pid")), "steps": [{k: x.get(k) for k in ("id", "status", "attempts", "reason")}
                                                                    for x in s.get("steps", [])]})
        print(json.dumps({"daemon_running": daemon_running(home), "campaigns": out}, indent=2))
        return 0
    if a.cmd == "stop":
        p = home / "campaigns" / a.campaign / "state.json"
        pid = json.loads(p.read_text()).get("pid") if p.exists() else None
        if pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
            print(f"stop signal sent to conductor {pid}; the current daemon job finishes or times out, no new step starts")
        else:
            print("conductor not running")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
