#!/usr/bin/env python3
"""MISSION-001 test suite for sanctuary_daemon.py (criteria C1-C7 in ../protocol.md).

Dev:      python3 test_daemon.py            (small trial counts, instrument check)
Confirm:  python3 test_daemon.py --confirm  (frozen trial counts; writes ../evidence/confirm_v2_results.json)
No model is called. Every test uses a throwaway --home directory and the shell runner.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAEMON = HERE / "sanctuary_daemon.py"
CONFIRM = "--confirm" in sys.argv
N_DETECT = 20 if CONFIRM else 3
N_TIMEOUT = 5 if CONFIRM else 1
IDLE_SEC = 15 if CONFIRM else 4
RESULTS: dict = {"mode": "confirm" if CONFIRM else "dev"}


def write_job(home: Path, jid: str, runner: dict, **extra) -> tuple[Path, float]:
    q = home / "queue"
    q.mkdir(parents=True, exist_ok=True)
    job = {"id": jid, "mission_id": "MISSION-001", "status": "QUEUED", "authorized_by": "test",
           "created_at": time.time(), "timeout_sec": 30, "runner": runner, **extra}
    tmp = q / f".{jid}.tmp"
    tmp.write_text(json.dumps(job))
    t0 = time.time()
    os.replace(tmp, q / f"{jid}.json")
    return q / f"{jid}.json", t0


def shell(cmd: str) -> dict:
    return {"type": "shell", "argv": ["sh", "-c", cmd]}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def wait_for(pred, timeout=20.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return None


class DaemonCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([sys.executable, str(DAEMON), "--home", str(self.home), "run"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertTrue(wait_for(lambda: (self.home / "events.jsonl").exists()
                                 and "daemon_start" in (self.home / "events.jsonl").read_text()))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=15)

    def tearDown(self):
        self.stop()
        self.tmp.cleanup()

    def events(self, kind):
        p = self.home / "events.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if json.loads(l)["event"] == kind] if p.exists() else []

    def state(self):
        return json.loads((self.home / "state.json").read_text())

    # C1
    def test_c1_detection_latency(self):
        self.start()
        lat = []
        for i in range(N_DETECT):
            p, t0 = write_job(self.home, f"c1-{i:02d}", shell("true"))
            j = wait_for(lambda: load(p) if load(p).get("status") == "COMPLETED" else None)
            self.assertIsNotNone(j, f"job c1-{i} did not complete")
            lat.append(j["running_at"] - t0)
            time.sleep(0.3)  # let the daemon return to idle wait between trials
        RESULTS["C1"] = {"n": len(lat), "max_s": max(lat), "median_s": sorted(lat)[len(lat) // 2],
                         "pass": all(x < 2.0 for x in lat)}
        self.assertTrue(RESULTS["C1"]["pass"], lat)

    # C2
    def test_c2_timeout_sigkill(self):
        # Amendment v2: a fresh daemon per trial. With one daemon, the 3rd consecutive timeout
        # correctly opens the circuit breaker (C4) and holds trials 4-5 in QUEUED.
        trials = []
        for i in range(N_TIMEOUT):
            self.stop()
            self.tmp.cleanup()
            self.tmp = tempfile.TemporaryDirectory()
            self.home = Path(self.tmp.name)
            self.start()
            p, _ = write_job(self.home, f"c2-{i}", shell("sleep 60 & sleep 60; wait"), timeout_sec=2)
            j = wait_for(lambda: load(p) if load(p).get("status") not in ("QUEUED", "RUNNING") else None, 20)
            self.assertIsNotNone(j)
            alive = True
            try:
                os.killpg(j["pgid"], 0)
            except ProcessLookupError:
                alive = False
            trials.append({"status": j["status"], "elapsed_s": j["ended_at"] - j["running_at"],
                           "group_alive": alive})
        RESULTS["C2"] = {"trials": trials, "pass": all(t["status"] == "TIMED_OUT" and t["elapsed_s"] < 6
                                                       and not t["group_alive"] for t in trials)}
        self.assertTrue(RESULTS["C2"]["pass"], trials)

    # C3
    def test_c3_idle_zero_dispatch_low_cpu(self):
        self.start()
        time.sleep(1.0)
        cpu0 = self._cpu()
        time.sleep(IDLE_SEC)
        cpu1 = self._cpu()
        frac = (cpu1 - cpu0) / IDLE_SEC
        RESULTS["C3"] = {"idle_s": IDLE_SEC, "cpu_s": cpu1 - cpu0, "cpu_frac": frac,
                         "dispatches": len(self.events("dispatch")),
                         "pass": frac < 0.01 and not self.events("dispatch")}
        self.assertTrue(RESULTS["C3"]["pass"], RESULTS["C3"])

    def _cpu(self) -> float:
        out = subprocess.run(["ps", "-o", "time=", "-p", str(self.proc.pid)], capture_output=True, text=True).stdout.strip()
        mins, secs = out.split(":")
        return int(mins) * 60 + float(secs)

    # C4
    def test_c4_circuit_breaker(self):
        self.start()
        for i in range(3):
            p, _ = write_job(self.home, f"c4-fail-{i}", shell("exit 3"))
            self.assertTrue(wait_for(lambda: load(p).get("status") == "FAILED"))
        self.assertEqual(self.state()["daemon_status"], "BLOCKED_ON_HUMAN")
        p4, _ = write_job(self.home, "c4-next", shell("true"))
        time.sleep(3)
        held = load(p4)["status"] == "QUEUED"
        subprocess.run([sys.executable, str(DAEMON), "--home", str(self.home), "reset"], check=True, capture_output=True)
        # reset changes state.json only; nudge the watcher by touching the queue
        (self.home / "queue" / ".nudge").write_text("x")
        done = wait_for(lambda: load(p4).get("status") == "COMPLETED")
        RESULTS["C4"] = {"blocked_after_3": True, "held_while_blocked": held, "ran_after_reset": bool(done),
                         "pass": held and bool(done)}
        self.assertTrue(RESULTS["C4"]["pass"], RESULTS["C4"])

    # C5
    def test_c5_sequential_order_and_atomic_files(self):
        out = self.home / "order.txt"
        bad = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                for f in (self.home / "queue").glob("*.json"):
                    try:
                        json.loads(f.read_text())
                    except FileNotFoundError:
                        pass
                    except json.JSONDecodeError:
                        bad.append(f.name)
                time.sleep(0.005)

        paths = []
        for i in range(3):  # queue all before the daemon starts: order must come from created_at
            p, _ = write_job(self.home, f"c5-{i}", shell(f"echo {i} >> {out}; sleep 0.3"))
            paths.append(p)
            time.sleep(0.01)
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        self.start()
        ok = wait_for(lambda: all(load(p)["status"] == "COMPLETED" for p in paths))
        stop.set()
        order = out.read_text().split() if out.exists() else []
        RESULTS["C5"] = {"order": order, "unparseable_reads": len(bad),
                         "dispatch_count": len(self.events("dispatch")),
                         "pass": bool(ok) and order == ["0", "1", "2"] and not bad and len(self.events("dispatch")) == 3}
        self.assertTrue(RESULTS["C5"]["pass"], RESULTS["C5"])

    # C6
    def test_c6_gates(self):
        self.start()
        cases = {
            "c6-noauth": (shell("true"), {"authorized_by": ""}),
            "c6-copilot": ({"type": "cli-agent", "cli": "copilot", "instruction": "x", "cwd": str(self.home)}, {}),
            "c6-flag": ({"type": "cli-agent", "cli": "claude", "instruction": "x", "cwd": str(self.home),
                         "mode": "--dangerously-skip-permissions"}, {}),
            "c6-bypass": ({"type": "cli-agent", "cli": "claude", "instruction": "x", "cwd": str(self.home),
                           "isolated": False}, {}),
        }
        res = {}
        for jid, (runner, extra) in cases.items():
            p, _ = write_job(self.home, jid, runner, **extra)
            j = wait_for(lambda: load(p) if load(p).get("status") != "QUEUED" else None)
            res[jid] = {"status": j and j["status"], "reason": j and j.get("reason")}
        RESULTS["C6"] = {"cases": res, "dispatches": len(self.events("dispatch")),
                         "pass": all(v["status"] == "REJECTED" for v in res.values()) and not self.events("dispatch")}
        self.assertTrue(RESULTS["C6"]["pass"], RESULTS["C6"])

    # C7
    def test_c7_crash_recovery(self):
        marker = self.home / "marker"
        p, _ = write_job(self.home, "c7", shell(f"test -f {marker} && exit 0; sleep 30"), timeout_sec=60)
        self.start()
        j = wait_for(lambda: load(p) if load(p).get("status") == "RUNNING" else None)
        self.assertIsNotNone(j)
        pgid = j["pgid"]
        self.stop()
        try:
            os.killpg(pgid, 0)
            killed = False
        except ProcessLookupError:
            killed = True
        after_stop = load(p)["status"]
        marker.write_text("x")
        self.start()
        done = wait_for(lambda: load(p) if load(p).get("status") == "COMPLETED" else None)
        RESULTS["C7"] = {"group_killed_on_stop": killed, "status_after_stop": after_stop,
                         "attempts": done and done.get("attempts"),
                         "pass": killed and after_stop == "QUEUED" and bool(done) and done.get("attempts") == 2}
        self.assertTrue(RESULTS["C7"]["pass"], RESULTS["C7"])


if __name__ == "__main__":
    argv = [a for a in sys.argv if a != "--confirm"]
    prog = unittest.main(argv=argv, exit=False, verbosity=2)
    RESULTS["tests_run"] = prog.result.testsRun
    RESULTS["failures"] = len(prog.result.failures) + len(prog.result.errors)
    RESULTS["all_pass"] = RESULTS["failures"] == 0
    if CONFIRM:
        ev = HERE.parent / "evidence"
        ev.mkdir(exist_ok=True)
        (ev / "confirm_v2_results.json").write_text(json.dumps(RESULTS, indent=2))
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in RESULTS.items()}))
    sys.exit(0 if RESULTS["all_pass"] else 1)
