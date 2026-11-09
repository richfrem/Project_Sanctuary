#!/usr/bin/env python3
"""automations.py - registry, approval and scheduler for Sanctuary Bot (stdlib, any OS).

An automation is a definition file `<definitions_dir>/<name>.json` (tracked in the repo by
default: `.agent/automations/`). Approval is machine-local and bound to the definition's
content hash: an edited definition stops running until it is approved again.

Every run goes through conductor.py (a one-step `job` becomes a one-step pipeline), so every
run gets the same outcome checks. The daemon calls `Scheduler.tick()` while idle.

Machine-local state under <home>/automations/:
  approvals.json  {name: {sha256, approved_by, approved_at}}
  state.json      {name: {last_run_at, last_status, next_due, current_run, failed_runs, paused, ...}}
  <name>/history.jsonl   one line per finished run
  <home>/inbox.md        human-readable run results (newest last)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDUCTOR = HERE / "conductor.py"
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
DEFAULT_PAUSE_AFTER = 3
FINAL = {"COMPLETED", "PARTIAL", "STALLED", "EXHAUSTED", "BLOCKED_ON_HUMAN", "STOPPED"}


def read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def content_hash(defn: dict) -> str:
    return hashlib.sha256(json.dumps(defn, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_every(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([smhdw])\s*", str(text))
    if not m:
        raise ValueError(f"bad interval {text!r} (use e.g. 30m, 6h, 1d, 1w)")
    return int(m.group(1)) * UNITS[m.group(2)]


def next_due(schedule: dict, after: float) -> float | None:
    """First due time strictly after `after` (epoch seconds, local time for `at`). None = manual only."""
    if not schedule or schedule.get("manual"):
        return None
    every = parse_every(schedule["every"])
    at = schedule.get("at")
    if not at:
        return after + every
    if every % 86400:
        raise ValueError("`at` needs an interval in whole days (1d, 7d, 1w)")
    hh, mm = (int(x) for x in at.split(":"))
    base = dt.datetime.fromtimestamp(after)
    cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    while cand.timestamp() <= after:
        cand += dt.timedelta(days=1)
    return cand.timestamp()


def catch_up(schedule: dict, due: float, now: float) -> tuple[bool, float | None]:
    """Decide what to do when `due` <= now. Returns (run_now, next_due_after_decision)."""
    if schedule.get("catch_up", "once") == "skip":
        every = parse_every(schedule["every"])
        if now - due >= every:  # missed at least one whole slot: do not run, jump ahead
            return False, next_due(schedule, now)
    return True, next_due(schedule, now)


def to_campaign(name: str, defn: dict, run_id: str, approved_by: str, repo_root: Path, previous: str | None) -> dict:
    wt = Path(defn["worktree"])
    wt = wt if wt.is_absolute() else (repo_root / wt)
    m = {"id": run_id, "authorized_by": approved_by, "worktree": str(wt.resolve()),
         "engine": defn["engine"], "rails": defn.get("rails", {}), "mission_id": defn.get("mission_id"),
         "automation": name}
    if previous:
        m["previous_run"] = previous
    if "job" in defn:
        step = {"id": "job", **{k: v for k, v in defn["job"].items() if k in ("goal", "shell", "verify")}}
        m.update({"mode": "pipeline", "steps": [step]})
    else:
        m.update({"mode": defn.get("mode", "pipeline")})
        if "steps" in defn:
            m["steps"] = defn["steps"]
        if "sweep" in defn:
            m["sweep"] = defn["sweep"]
    return m


class Registry:
    def __init__(self, home: Path, definitions: Path, now=time.time):
        self.home, self.defs, self.now = home, definitions, now
        self.dir = home / "automations"
        self.approvals_p, self.state_p = self.dir / "approvals.json", self.dir / "state.json"
        self.repo_root = definitions.resolve().parents[1] if definitions.resolve().parent.name == ".agent" else definitions.resolve().parent

    # --- definitions ---
    def names(self) -> list[str]:
        return sorted(p.stem for p in self.defs.glob("*.json")) if self.defs.is_dir() else []

    def load(self, name: str) -> dict:
        return json.loads((self.defs / f"{name}.json").read_text())

    def validate(self, name: str, defn: dict | None = None) -> list[str]:
        sys.path.insert(0, str(HERE))
        import conductor
        try:
            defn = defn if defn is not None else self.load(name)
        except (OSError, json.JSONDecodeError) as e:
            return [f"cannot read definition: {e}"]
        errs = []
        if defn.get("name") != name:
            errs.append(f"`name` must equal the file name ({name})")
        if ("job" in defn) == ("steps" in defn or "sweep" in defn):
            errs.append("define either `job` (one step) or `steps`/`sweep` (several), not both or neither")
        try:
            next_due(defn.get("schedule", {"manual": True}), self.now())
        except (ValueError, KeyError) as e:
            errs.append(f"schedule: {e}")
        if not defn.get("worktree") or not defn.get("engine"):
            errs.append("needs `worktree` and `engine`")
            return errs
        errs += [e for e in conductor.validate(to_campaign(name, defn, "validate", "validator", self.repo_root, None))
                 if not e.startswith("missing authorized_by")]
        return errs

    # --- approval ---
    def approvals(self) -> dict:
        return read_json(self.approvals_p, {})

    def approve(self, name: str, by: str) -> dict:
        defn = self.load(name)
        errs = self.validate(name, defn)
        if errs:
            raise ValueError("; ".join(errs))
        a = self.approvals()
        a[name] = {"sha256": content_hash(defn), "approved_by": by, "approved_at": self.now()}
        write_json(self.approvals_p, a)
        st = self.state()
        s = st.setdefault(name, {})
        s.update({"paused": False, "pause_reason": None, "failed_runs": 0,
                  "next_due": next_due(defn.get("schedule", {"manual": True}), self.now())})
        write_json(self.state_p, st)
        return a[name]

    def approval_status(self, name: str) -> str:
        a = self.approvals().get(name)
        if not a:
            return "NOT_APPROVED"
        try:
            return "APPROVED" if a["sha256"] == content_hash(self.load(name)) else "CHANGED_SINCE_APPROVAL"
        except (OSError, json.JSONDecodeError):
            return "DEFINITION_MISSING"

    # --- state ---
    def state(self) -> dict:
        return read_json(self.state_p, {})

    def set(self, name: str, **kw) -> None:
        st = self.state()
        st.setdefault(name, {}).update(kw)
        write_json(self.state_p, st)

    def history(self, name: str, limit: int = 20) -> list[dict]:
        p = self.dir / name / "history.jsonl"
        lines = p.read_text().splitlines() if p.exists() else []
        return [json.loads(l) for l in lines[-limit:]]


class Scheduler:
    """Called by the daemon while idle. Finishes running runs, then launches due approved runs."""

    def __init__(self, home: Path, config: dict, now=time.time, launch=None):
        self.home, self.cfg, self.now = home, config, now
        self.reg = Registry(home, Path(config["definitions_dir"]), now)
        self.launch = launch or self._spawn_conductor

    def tick(self) -> list[str]:
        events = []
        for name in self.reg.names():
            s = self.reg.state().get(name, {})
            if s.get("current_run"):
                events += self._reconcile(name, s)
                continue
            if s.get("paused"):
                continue
            status = self.reg.approval_status(name)
            if status != "APPROVED":
                if s.get("blocked_reason") != status:
                    self.reg.set(name, blocked_reason=status)
                continue
            defn = self.reg.load(name)
            sched = defn.get("schedule", {"manual": True})
            due = s.get("next_due")
            forced = s.get("run_now")
            if not forced and (due is None or due > self.now()):
                continue
            run, nxt = (True, next_due(sched, self.now())) if forced else catch_up(sched, due, self.now())
            if not run:
                self.reg.set(name, next_due=nxt, skipped_at=self.now())
                events.append(f"{name}: skipped missed slot")
                continue
            prev = (self.reg.history(name, 1) or [{}])[-1].get("summary")
            run_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.now()))}"
            camp = to_campaign(name, defn, run_id, self.reg.approvals()[name]["approved_by"], self.reg.repo_root, prev)
            pid = self.launch(camp)
            self.reg.set(name, current_run=run_id, current_pid=pid, run_started_at=self.now(), next_due=nxt,
                         run_now=False, blocked_reason=None)
            events.append(f"{name}: started {run_id}")
        return events

    def _spawn_conductor(self, camp: dict) -> int:
        d = self.home / "campaigns" / camp["id"]
        write_json(d / "manifest.json", camp)
        log = open(d / "conductor.out", "ab")
        p = subprocess.Popen([sys.executable, str(CONDUCTOR), "--home", str(self.home), "run", camp["id"]],
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        return p.pid

    def _reconcile(self, name: str, s: dict) -> list[str]:
        run_id = s["current_run"]
        st = read_json(self.home / "campaigns" / run_id / "state.json", {})
        status = st.get("status")
        alive = _alive(s.get("current_pid"))
        if status in FINAL or (not alive and status != "RUNNING" and self.now() - s.get("run_started_at", 0) > 30):
            status = status if status in FINAL else "CRASHED"
        elif not alive and status == "RUNNING":
            status = "CRASHED"
        else:
            return []
        ok = status == "COMPLETED"
        last = (st.get("history") or [{}])[-1]
        summary = (f"{run_id} ended {status}; last step {last.get('step')}: {last.get('reason')}; "
                   f"commit {last.get('commit')}; files {', '.join((last.get('files') or [])[:10]) or 'none'}")
        rec = {"run": run_id, "status": status, "ended_at": self.now(), "cycles": st.get("cycles"), "summary": summary}
        hp = self.reg.dir / name / "history.jsonl"
        hp.parent.mkdir(parents=True, exist_ok=True)
        with open(hp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        failed = 0 if ok else s.get("failed_runs", 0) + 1
        limit = int(self.reg.load(name).get("pause_after_failed_runs", DEFAULT_PAUSE_AFTER)) if (self.reg.defs / f"{name}.json").exists() else DEFAULT_PAUSE_AFTER
        upd = {"current_run": None, "current_pid": None, "last_run_at": self.now(), "last_status": status, "failed_runs": failed}
        if failed >= limit:
            upd.update(paused=True, pause_reason=f"{failed} failed runs in a row")
        self.reg.set(name, **upd)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.now()))
        line = f"- {when} **{name}** {status}" + (" (paused: " + upd["pause_reason"] + ")" if upd.get("paused") else "") + f": {summary}\n"
        with open(self.home / "inbox.md", "a") as f:
            f.write(line)
        self._notify(name, status, summary)
        return [f"{name}: {status}"]

    def _notify(self, name: str, status: str, summary: str) -> None:
        cmd = self.cfg.get("notify_cmd")
        if not cmd:
            return
        env = {**os.environ, "BOT_AUTOMATION": name, "BOT_STATUS": status, "BOT_SUMMARY": summary}
        try:
            subprocess.run(cmd, env=env, timeout=30, capture_output=True)  # argv list, no shell
        except (OSError, subprocess.TimeoutExpired):
            pass


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def load_config(home: Path) -> dict | None:
    cfg = read_json(home / "config.json", None)
    if cfg and cfg.get("definitions_dir"):
        return cfg
    return None
