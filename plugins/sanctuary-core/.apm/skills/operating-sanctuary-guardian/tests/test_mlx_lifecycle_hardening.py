"""MISSION-027 tests: MLX server lifecycle hardening (discovery, teardown, memory), using a stub process tree.

Purpose: prove that port discovery ignores client sockets, that stop_server tears down the launcher AND its listening worker
    (no zombie listener), and that status/memory describe the worker. Uses a stub "launcher -> worker" tree, no model, no network.
Key Input Dependencies: pytest; scripts/mlx_server_manager.py; lsof and ps (macOS/Linux).
Key Functions: test_discovery_ignores_client_sockets, test_stop_tears_down_worker_child, test_stop_reports_failure_if_worker_survives,
    test_status_describes_worker_not_launcher, test_twenty_cycles_leave_nothing_behind, test_unsafe_group_falls_back_to_tree_kill,
    test_stale_state_pid_is_never_signalled, test_parse_footprint, test_group_safety_rules, test_reap_only_managed,
    test_wakeup_reaps_idle_managed_server_only, test_live_real_model_lifecycle (gated).
Usage: python3 -m pytest plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_mlx_lifecycle_hardening.py
"""
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import mlx_server_manager as msm  # noqa: E402

# The stub's file name contains "mlx_lm" so it matches the manager's process-signature safety check, like the real server.
STUB = textwrap.dedent('''
    import http.server, signal, subprocess, sys
    mode, port = sys.argv[1], int(sys.argv[2])
    ignore = "--ignore-term" in sys.argv
    if mode == "launcher":
        child = subprocess.Popen([sys.executable, __file__, "worker", str(port)] + (["--ignore-term"] if ignore else []))
        child.wait()
    else:
        if ignore:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')
            def log_message(self, *a): pass
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    out = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def wait_until(cond, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Start a stub server tree (launcher in its own session + listening worker); yield a factory; clean up afterwards."""
    monkeypatch.chdir(tmp_path)
    stub = tmp_path / "mlx_lm_stub_server.py"
    stub.write_text(STUB)
    started = []

    def make(ignore_term: bool = False):
        port = free_port()
        cmd = [sys.executable, str(stub), "launcher", str(port)] + (["--ignore-term"] if ignore_term else [])
        launcher = subprocess.Popen(cmd, start_new_session=True)
        assert wait_until(lambda: msm.is_server_responding("127.0.0.1", port)), "stub server did not come up"
        worker = int(subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout.split()[0])
        msm.save_state({"status": "running", "pid": launcher.pid, "port": port, "started_at": time.time(), "last_active": time.time()})
        started.append((launcher, worker))
        return port, launcher, worker

    yield make
    for launcher, worker in started:
        for pid in (worker, launcher.pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        launcher.wait(timeout=5)


def test_discovery_ignores_client_sockets(tree):
    port, launcher, worker = tree()
    client = socket.create_connection(("127.0.0.1", port))  # this test process now owns a client socket on the port
    try:
        assert msm.get_pid_listening_on_port(port) == worker, "discovery returned a client PID instead of the listener"
    finally:
        client.close()


def test_stop_tears_down_worker_child(tree):
    port, launcher, worker = tree()
    result = msm.stop_server(port=port, force=False)
    assert result["status"] == "stopped", result
    assert wait_until(lambda: not alive(worker)), "worker child still alive after stop"
    assert wait_until(lambda: not alive(launcher.pid)), "launcher still alive after stop"
    assert not msm.is_server_responding("127.0.0.1", port), "something still listens on the port after stop"


def test_stop_reports_failure_if_worker_survives(tree):
    port, launcher, worker = tree(ignore_term=True)  # the worker ignores SIGTERM
    result = msm.stop_server(port=port, force=False)
    assert result["status"] == "error", f"stop claimed success while the worker survived: {result}"
    forced = msm.stop_server(port=port, force=True)
    assert forced["status"] == "stopped", forced
    assert wait_until(lambda: not alive(worker)), "worker survived a forced stop"


def test_status_describes_worker_not_launcher(tree):
    port, launcher, worker = tree()
    status = msm.get_status(port=port)
    assert status["pid"] == worker, f"status reports pid {status['pid']} (launcher {launcher.pid}), expected the listening worker {worker}"
    assert "footprint_mb" in status


def test_twenty_cycles_leave_nothing_behind(tree):
    """Pre-registered prediction 2: 20 start/stop cycles, 0 survivors, 0 client PIDs returned by discovery."""
    for i in range(20):
        port, launcher, worker = tree()
        client = socket.create_connection(("127.0.0.1", port))
        try:
            assert os.getpid() not in msm.get_listening_pids(port), f"cycle {i}: a client PID was returned"
            assert msm.get_pid_listening_on_port(port) == worker, f"cycle {i}: wrong listener"
        finally:
            client.close()
        result = msm.stop_server(port=port)
        assert result["status"] == "stopped" and result["verified"], f"cycle {i}: {result}"
        assert wait_until(lambda: not alive(worker) and not alive(launcher.pid)), f"cycle {i}: survivor"
        assert not msm.get_listening_pids(port), f"cycle {i}: port still has a listener"


def test_unsafe_group_falls_back_to_tree_kill(tmp_path, monkeypatch):
    """Prediction 4: a server launched from a shell script shares its group with the script and a bystander.
    The group must NOT be signalled as a unit; only the server tree dies and the script and bystander survive."""
    monkeypatch.chdir(tmp_path)
    stub = tmp_path / "mlx_lm_stub_server.py"
    stub.write_text(STUB)
    port = free_port()
    script = tmp_path / "run_server.sh"  # its own command line contains no "mlx_lm", like a user's launcher script
    script.write_text(f"#!/bin/bash\n{sys.executable} {stub} launcher {port} &\nsleep 60 &\nwait\n")
    shell = subprocess.Popen(["bash", str(script)], start_new_session=True)
    try:
        assert wait_until(lambda: msm.is_server_responding("127.0.0.1", port)), "stub server did not come up"
        table = msm._ps_table()
        worker = msm.get_listening_pids(port)[0]
        launcher = table[worker]["ppid"]
        bystander = next(p for p, i in table.items() if i["ppid"] == shell.pid and i["command"].startswith("sleep"))
        assert table[worker]["pgid"] == shell.pid and not msm._group_is_safe(shell.pid, table)
        msm.save_state({"status": "running", "pid": launcher, "port": port, "started_at": time.time(), "last_active": time.time()})
        result = msm.stop_server(port=port)
        assert result["status"] == "stopped" and result["signalled"] == "process_tree", result
        assert wait_until(lambda: not alive(worker) and not alive(launcher)), "server tree survived"
        assert alive(bystander), "an unrelated process in the same group was killed"
        assert alive(shell.pid), "the launching shell script was killed"
    finally:
        try:
            os.killpg(shell.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        shell.wait(timeout=5)


def test_stale_state_pid_is_never_signalled(tmp_path, monkeypatch):
    """A state-file PID that no longer belongs to an MLX process (reused PID) must not be killed."""
    monkeypatch.chdir(tmp_path)
    innocent = subprocess.Popen(["sleep", "60"])
    try:
        msm.save_state({"status": "running", "pid": innocent.pid, "port": free_port()})
        result = msm.stop_server(port=free_port())
        assert result["status"] == "not_running", result
        assert alive(innocent.pid), "stop_server signalled a non-MLX process from a stale state file"
    finally:
        innocent.kill()
        innocent.wait(timeout=5)


def test_parse_footprint():
    assert msm._parse_footprint("python [74300]: 64-bit    Footprint: 4711 MB (16384 bytes per page)") == 4711.0
    assert msm._parse_footprint("Footprint: 4.5 GB") == 4608.0
    assert msm._parse_footprint("Footprint: 2048 KB") == 2.0
    assert msm._parse_footprint("no such process") is None


def test_footprint_falls_back_to_zero_when_tool_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("footprint")
    monkeypatch.setattr(msm.subprocess, "run", boom)
    assert msm.get_process_footprint_mb(1) == 0.0


def test_group_safety_rules(monkeypatch):
    mlx = {"ppid": 1, "pgid": 500, "stat": "S", "command": "/x/bin/python /x/bin/mlx_lm.server --port 8080"}
    uvx = {"ppid": 1, "pgid": 500, "stat": "S", "command": "/opt/homebrew/bin/uv tool uvx --from mlx-lm mlx_lm.server"}
    other = {"ppid": 1, "pgid": 500, "stat": "S", "command": "/bin/zsh -il"}
    zombie = {"ppid": 1, "pgid": 500, "stat": "Z", "command": "(defunct)"}
    assert msm._group_is_safe(500, {10: mlx, 11: uvx, 12: zombie})
    assert not msm._group_is_safe(500, {10: mlx, 13: other})
    assert not msm._group_is_safe(1, {10: mlx})
    assert not msm._group_is_safe(999, {})
    monkeypatch.setattr(os, "getpgrp", lambda: 500)
    assert not msm._group_is_safe(500, {10: mlx}), "our own process group must never be signalled"


def test_reap_only_managed(monkeypatch):
    calls = []
    monkeypatch.setattr(msm, "stop_server", lambda port, force: calls.append(port) or {"status": "stopped", "reclaimed_rss_mb": 4700.0})
    base = {"is_active": True, "idle_sec": 5000.0, "idle_timeout_sec": 900, "pid": 42}
    monkeypatch.setattr(msm, "get_status", lambda host, port: {**base, "managed": False})
    assert msm.reap_if_idle(port=8080, only_managed=True)["status"] == "unmanaged" and not calls
    monkeypatch.setattr(msm, "get_status", lambda host, port: {**base, "managed": True})
    assert msm.reap_if_idle(port=8080, only_managed=True)["status"] == "reaped" and calls == [8080]


def test_wakeup_reaps_idle_managed_server_only(monkeypatch):
    """Protocol 128 boot timer: wakeup calls reap_if_idle(only_managed=True) and reports a reap; failures never break wakeup."""
    import guardian_wakeup as gw
    seen = {}
    monkeypatch.setattr(msm, "reap_if_idle", lambda **kw: seen.update(kw) or {"status": "reaped", "idle_sec": 2000.0, "reclaimed_rss_mb": 4300.0})
    checks = gw.check_runtime_telemetry()
    assert seen == {"only_managed": True}
    server = next(c for c in checks if "MLX Server" in c["component"])
    assert "Reaped an idle managed server" in server["description"]
    monkeypatch.setattr(msm, "reap_if_idle", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert any("MLX Server" in c["component"] for c in gw.check_runtime_telemetry())  # still reports


# ---- live validation with the real model (gated) -----------------------------------------------------------------
LIVE_PORT = 8099
LIVE_RESULTS = Path(__file__).resolve().parents[6] / "LEARNING" / "research" / "experiments" / "mission-027" / "live_results.json"


def _free_memory_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.split(":")[1].strip().rstrip("%"))
    return 0


def _warm_up(port: int) -> None:
    """One small generation so the weights are paged in (the model loads lazily; an idle fresh server under-reports)."""
    import urllib.request
    body = json.dumps({"model": msm.DEFAULT_MODEL, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 32}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


@pytest.mark.skipif(os.environ.get("RUN_LIVE_MLX") != "1", reason="set RUN_LIVE_MLX=1 (loads the 4-bit 7B model on port 8099, ~4.7 GB)")
def test_live_real_model_lifecycle(tmp_path, monkeypatch):
    """Pre-registered prediction 3 on a real model, on a spare port (the server on :8080 is never touched).

    Two measurement points are recorded: `fresh` (immediately after boot; run 1 showed this under-reports because loading is lazy, so the
    original >= 4000 MB prediction MISSED there) and `warm` (after one 32-token generation). Assertions: warm footprint >= 4000 MB,
    no client PID (the fresh reading is recorded but not asserted: it varied 562-3,317 MB across runs), verified stop, no survivors, memory returned to the system.
    """
    if _free_memory_pct() < 30:
        pytest.skip("less than 30% free memory; not loading a second model")
    monkeypatch.chdir(tmp_path)
    cycles = int(os.environ.get("LIVE_CYCLES", "2"))
    rows = []
    for i in range(cycles):
        free_before = _free_memory_pct()
        started = msm.start_server(port=LIVE_PORT, timeout_sec=240)
        assert started["status"] == "started", started
        worker, launcher = started["worker_pid"], started["pid"]
        fresh = msm.get_status(port=LIVE_PORT)
        _warm_up(LIVE_PORT)
        client = socket.create_connection(("127.0.0.1", LIVE_PORT))
        try:
            listeners = msm.get_listening_pids(LIVE_PORT)
            warm = msm.get_status(port=LIVE_PORT)
        finally:
            client.close()
        free_running = _free_memory_pct()
        stopped = msm.stop_server(port=LIVE_PORT)
        gone = wait_until(lambda: not alive(worker) and not alive(launcher), timeout=15)
        time.sleep(1.0)
        row = {"cycle": i + 1, "boot_latency_sec": started["boot_latency_sec"], "worker_pid": worker, "launcher_pid": launcher,
               "listeners_while_running": listeners, "client_pid_returned": os.getpid() in listeners,
               "fresh_footprint_mb": fresh["footprint_mb"], "fresh_ps_rss_mb": fresh["rss_mb"],
               "warm_footprint_mb": warm["footprint_mb"], "warm_ps_rss_mb": warm["rss_mb"], "status_pid_is_worker": warm["pid"] == worker,
               "original_prediction_fresh_ge_4000": fresh["footprint_mb"] >= 4000,
               "stop_status": stopped["status"], "signalled": stopped.get("signalled"), "verified": stopped.get("verified"),
               "reclaimed_footprint_mb": stopped.get("reclaimed_footprint_mb"), "processes_gone": gone,
               "listeners_after": msm.get_listening_pids(LIVE_PORT), "free_pct_before_running_after": [free_before, free_running, _free_memory_pct()]}
        rows.append(row)
        print(json.dumps(row))
    LIVE_RESULTS.write_text(json.dumps(rows, indent=2))
    for r in rows:
        assert r["warm_footprint_mb"] >= 4000, r
        assert not r["client_pid_returned"] and r["status_pid_is_worker"], r
        assert r["stop_status"] == "stopped" and r["verified"] and r["processes_gone"] and not r["listeners_after"], r
        assert r["free_pct_before_running_after"][2] >= r["free_pct_before_running_after"][1] + 10, r


def test_default_idle_timeout_is_at_most_five_minutes(monkeypatch):
    """MISSION-029: measured cold start p95 4.97 s and 3.6 GiB returned per reap, so the default reap threshold is 300 s."""
    import importlib
    monkeypatch.delenv("MLX_IDLE_TIMEOUT_SEC", raising=False)
    fresh = importlib.reload(msm)
    try:
        assert fresh.DEFAULT_IDLE_TIMEOUT <= 300
    finally:
        monkeypatch.setenv("MLX_IDLE_TIMEOUT_SEC", "900")
        importlib.reload(msm)
        monkeypatch.delenv("MLX_IDLE_TIMEOUT_SEC", raising=False)
        importlib.reload(msm)


def test_default_reap_leaves_unmanaged_server_alone_even_with_tiny_timeout(monkeypatch):
    """MISSION-029 finding: an unmanaged server reports idle_sec 0.0, so even `reap --timeout 1` with default flags cannot stop it."""
    calls = []
    monkeypatch.setattr(msm, "stop_server", lambda port, force: calls.append(port) or {"status": "stopped"})
    monkeypatch.setattr(msm, "get_status", lambda host, port: {"is_active": True, "managed": False, "idle_sec": 0.0,
                                                               "idle_timeout_sec": 900, "pid": 42})
    res = msm.reap_if_idle(port=8080, idle_timeout_sec=1)
    assert res["status"] == "active" and not calls
