#!/usr/bin/env python3
"""Tests for automations.py (registry, approval, scheduler) and bot.py (end to end). No model calls.

Run: python3 test_bot.py
"""
import datetime as dt
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOT = HERE / "bot.py"
sys.path.insert(0, str(HERE))
import automations as A  # noqa: E402


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def git_repo(base: Path) -> tuple[Path, Path]:
    repo = base / "repo"
    repo.mkdir()
    for c in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
              ["commit", "-q", "--allow-empty", "-m", "init"]):
        subprocess.run(["git", "-C", str(repo), *c], check=True, capture_output=True)
    wt = repo / ".worktrees" / "task"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "-b", "task", str(wt)], check=True, capture_output=True)
    defs = repo / ".agent" / "automations"
    defs.mkdir(parents=True)
    return repo, defs


def job_def(name, schedule, shell="sh -c 'date > t.txt && git add t.txt && git commit -qm tick'", **kw):
    return {"name": name, "schedule": schedule, "worktree": ".worktrees/task", "engine": {"cli": "shell"},
            "job": {"shell": shell}, **kw}


class ScheduleMath(unittest.TestCase):
    def test_interval(self):
        self.assertEqual(A.next_due({"every": "30m"}, 1000), 1000 + 1800)

    def test_daily_at(self):
        base = dt.datetime(2026, 11, 9, 8, 0).timestamp()  # after 07:00 -> tomorrow 07:00
        self.assertEqual(dt.datetime.fromtimestamp(A.next_due({"every": "1d", "at": "07:00"}, base)),
                         dt.datetime(2026, 11, 10, 7, 0))
        early = dt.datetime(2026, 11, 9, 6, 0).timestamp()
        self.assertEqual(dt.datetime.fromtimestamp(A.next_due({"every": "1d", "at": "07:00"}, early)),
                         dt.datetime(2026, 11, 9, 7, 0))

    def test_manual_and_bad(self):
        self.assertIsNone(A.next_due({"manual": True}, 0))
        with self.assertRaises(ValueError):
            A.next_due({"every": "soon"}, 0)
        with self.assertRaises(ValueError):
            A.next_due({"every": "6h", "at": "07:00"}, 0)

    def test_catch_up(self):
        s_once, s_skip = {"every": "1h"}, {"every": "1h", "catch_up": "skip"}
        self.assertTrue(A.catch_up(s_once, 0, 3 * 3600)[0])          # missed 3 slots: run once
        self.assertFalse(A.catch_up(s_skip, 0, 3 * 3600)[0])         # skip: don't run, jump ahead
        self.assertTrue(A.catch_up(s_skip, 0, 60)[0])                # just due: run


class SchedulerUnit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name).resolve()
        self.repo, self.defs = git_repo(base)
        self.home = base / "home"
        self.clock = Clock(1_000_000.0)
        self.launched = []
        self.cfg = {"definitions_dir": str(self.defs)}

    def tearDown(self):
        self.tmp.cleanup()

    def sched(self, cfg=None):
        return A.Scheduler(self.home, cfg or self.cfg, now=self.clock, launch=lambda c: self.launched.append(c) or 99999999)

    def put(self, d):
        (self.defs / f"{d['name']}.json").write_text(json.dumps(d))

    def finish(self, run_id, status, step="job", commit="tick (abc)"):
        A.write_json(self.home / "campaigns" / run_id / "state.json",
                     {"status": status, "cycles": 1, "history": [{"step": step, "reason": "ok", "commit": commit, "files": ["t.txt"]}]})

    def test_unapproved_and_tampered_never_run(self):
        self.put(job_def("a", {"every": "1h"}))
        s = self.sched()
        self.clock.t += 7200
        s.tick()
        self.assertEqual(self.launched, [])                       # not approved
        A.Registry(self.home, self.defs, self.clock).approve("a", "tester")
        self.put(job_def("a", {"every": "1h"}, shell="sh -c 'rm -rf ~'"))  # edited after approval
        self.clock.t += 7200
        s.tick()
        self.assertEqual(self.launched, [])
        self.assertEqual(A.Registry(self.home, self.defs).approval_status("a"), "CHANGED_SINCE_APPROVAL")

    def test_due_run_overlap_memory_inbox_notify(self):
        out = Path(self.tmp.name) / "notified.txt"
        cfg = {**self.cfg, "notify_cmd": [sys.executable, "-c",
                                          f"import os; open({str(out)!r},'a').write(os.environ['BOT_AUTOMATION']+' '+os.environ['BOT_STATUS']+'\\n')"]}
        self.put(job_def("a", {"every": "1h"}))
        reg = A.Registry(self.home, self.defs, self.clock)
        reg.approve("a", "tester")
        s = self.sched(cfg)
        s.tick()
        self.assertEqual(self.launched, [])                       # not due yet
        self.clock.t += 3600
        s.tick()
        self.assertEqual(len(self.launched), 1)
        run1 = self.launched[0]["id"]
        self.assertEqual(self.launched[0]["authorized_by"], "tester")
        self.clock.t += 7200
        s.tick()                                                   # still running (no state yet, pid dead but <30 s? no: >30 s)
        # the fake pid is not alive and no state file -> treated as CRASHED after 30 s
        self.assertEqual(reg.state()["a"]["last_status"], "CRASHED")
        self.clock.t += 3600
        s.tick()
        run2 = self.launched[-1]["id"]
        self.assertNotEqual(run1, run2)
        self.assertIn("CRASHED", self.launched[-1]["previous_run"])  # cross-run memory
        self.finish(run2, "COMPLETED")
        s.tick()
        self.assertEqual(reg.state()["a"]["last_status"], "COMPLETED")
        self.assertIn("a COMPLETED", out.read_text())
        self.assertIn("**a** COMPLETED", (self.home / "inbox.md").read_text())

    def test_overlap_skip_while_running(self):
        self.put(job_def("a", {"every": "1m"}))
        A.Registry(self.home, self.defs, self.clock).approve("a", "tester")
        s = A.Scheduler(self.home, self.cfg, now=self.clock, launch=lambda c: self.launched.append(c) or __import__("os").getpid())
        for _ in range(5):                                         # live pid, no final state: run in progress
            self.clock.t += 120
            s.tick()
        self.assertEqual(len(self.launched), 1)

    def test_auto_pause_after_failed_runs_and_resume_rules(self):
        self.put(job_def("a", {"every": "1m"}, pause_after_failed_runs=2))
        reg = A.Registry(self.home, self.defs, self.clock)
        reg.approve("a", "tester")
        s = self.sched()
        for _ in range(2):
            self.clock.t += 120
            s.tick()
            self.finish(self.launched[-1]["id"], "PARTIAL")
            s.tick()
        st = reg.state()["a"]
        self.assertTrue(st["paused"])
        self.assertIn("2 failed runs", st["pause_reason"])
        self.clock.t += 600
        s.tick()
        self.assertEqual(len(self.launched), 2)                    # paused: nothing new

    def test_run_now_manual(self):
        self.put(job_def("m", {"manual": True}))
        reg = A.Registry(self.home, self.defs, self.clock)
        reg.approve("m", "tester")
        s = self.sched()
        self.clock.t += 10 ** 6
        s.tick()
        self.assertEqual(self.launched, [])
        reg.set("m", run_now=True)
        s.tick()
        self.assertEqual(len(self.launched), 1)

    def test_validation(self):
        reg = A.Registry(self.home, self.defs, self.clock)
        (self.defs / "bad.json").write_text(json.dumps({**job_def("bad", {"every": "1h"}), "name": "other"}))
        self.assertTrue(any("name" in e for e in reg.validate("bad")))
        self.put({**job_def("cp", {"every": "1h"}), "engine": {"cli": "copilot"}})
        self.assertTrue(any("copilot" in e for e in reg.validate("cp")))
        with self.assertRaises(ValueError):
            reg.approve("cp", "tester")


class EndToEnd(unittest.TestCase):
    def test_bot_cli_schedules_real_runs(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t).resolve()
            repo, defs = git_repo(base)
            home = base / "home"
            bot = [sys.executable, str(BOT), "--home", str(home)]
            ctx = "sh -c 'cp {context_file} ctx.txt; date >> log.txt; git add -A; git commit -qm tick'"
            (defs / "ticker.json").write_text(json.dumps(job_def("ticker", {"every": "2s"}, shell=ctx)))
            run = lambda *a: subprocess.run(bot + list(a), capture_output=True, text=True, cwd=repo)
            self.assertEqual(run("setup", "--definitions", str(defs)).returncode in (0, 1), True)
            self.assertEqual(run("approve", "ticker", "--by", "tester").returncode, 0)
            self.assertEqual(run("start").returncode, 0)
            try:
                reg = A.Registry(home, defs)
                ok = False
                for _ in range(120):
                    h = reg.history("ticker")
                    if len([x for x in h if x["status"] == "COMPLETED"]) >= 2:
                        ok = True
                        break
                    time.sleep(0.25)
                self.assertTrue(ok, run("status").stdout)
                ctxt = (repo / ".worktrees" / "task" / "ctx.txt").read_text()
                self.assertIn("PREVIOUS RUN OF THIS AUTOMATION", ctxt)
                status = run("status").stdout
                self.assertIn("ticker", status)
                self.assertIn("running", status)
                self.assertEqual(run("pause", "ticker").returncode, 0)
                self.assertIn("PAUSED", run("list").stdout)
            finally:
                run("stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
