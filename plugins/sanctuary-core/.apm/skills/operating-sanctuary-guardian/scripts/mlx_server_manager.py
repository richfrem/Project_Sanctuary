#!/usr/bin/env python3
"""mlx_server_manager.py - On-Demand MLX Local Server Lifecycle Controller.
Purpose: Manages background mlx_lm.server daemon lifecycle, health gates, and idle memory reclamation.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 Execution & Resource Management).
Key Inputs: Server host/port, MLX model repository, idle timeout (seconds), action command.
Key Outputs: Process PID, operational status, memory telemetry (RSS MB), and JSON lifecycle events.
Usage:
    python3 mlx_server_manager.py start [--port 8080] [--idle-timeout 300]
    python3 mlx_server_manager.py stop [--force]
    python3 mlx_server_manager.py status
    python3 mlx_server_manager.py ensure
    python3 mlx_server_manager.py reap
Key Functions: start_server(), stop_server(), get_status(), ensure_ready(), reap_if_idle(), touch_activity(),
    get_listening_pids(), get_process_footprint_mb(), _ps_table(), _group_is_safe(), _terminate().
Hardening (MISSION-027): discovery lists only LISTEN sockets (never client sockets); stop tears down the whole server tree (launcher
    + worker) and verifies nothing survives; memory is the worker's `footprint` (Metal/unified memory), not `ps` RSS, which reads ~10 MB
    for a worker holding a 4.7 GB model; idle reaping can be limited to servers this manager started.
"""

import os
import sys
import json
import time
import signal
import shutil
import urllib.request
import urllib.error
import subprocess
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

DEFAULT_HOST = os.environ.get("MLX_SERVER_HOST_IP", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MLX_SERVER_PORT", "8080"))
DEFAULT_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
DEFAULT_IDLE_TIMEOUT = int(os.environ.get("MLX_IDLE_TIMEOUT_SEC", "300"))  # 5 minutes (MISSION-029: cold start p95 4.97 s, 3.6 GiB returned)

STATE_DIR = Path(".agent/runtime")
STATE_FILE = STATE_DIR / "mlx_server.json"
FALLBACK_STATE_FILE = Path("/tmp/sanctuary_mlx_server.json")
LOG_FILE = STATE_DIR / "mlx_server.log"

# A process group is signalled as a unit only if every member looks like an MLX server process (see _group_is_safe).
MLX_PROC_SIGNATURE = re.compile(r"mlx_lm|mlx-lm|uvx|uv tool")
FOOTPRINT_RE = re.compile(r"Footprint:\s*([\d.,]+)\s*(KB|MB|GB)", re.IGNORECASE)


def _get_active_state_path() -> Path:
    """Resolve writable path for runtime state."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_FILE
    except Exception:
        return FALLBACK_STATE_FILE


def load_state() -> Dict[str, Any]:
    """Read server runtime state from disk."""
    path = _get_active_state_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    """Save server runtime state to disk."""
    path = _get_active_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to save mlx state to {path}: {e}\n")


def clear_state() -> None:
    """Remove runtime state file."""
    path = _get_active_state_path()
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def is_server_responding(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Probe if mlx_lm.server responds at /v1/models."""
    try:
        url = f"http://{host}:{port}/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "Sanctuary-Guardian-Lifecycle"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_listening_pids(port: int = DEFAULT_PORT) -> List[int]:
    """Return PIDs that LISTEN on the TCP port (client sockets connected to it are not included)."""
    try:
        cmd = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            return sorted({int(p) for p in res.stdout.split()})
    except Exception:
        pass
    return []


def get_pid_listening_on_port(port: int = DEFAULT_PORT) -> Optional[int]:
    """Find the process ID listening on the port (the worker), or None. Ignores client connections."""
    pids = get_listening_pids(port)
    return pids[0] if pids else None


def _ps_table() -> Dict[int, Dict[str, Any]]:
    """Snapshot of the process table: pid -> {ppid, pgid, stat, command}."""
    table: Dict[int, Dict[str, Any]] = {}
    try:
        res = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,stat=,command="], capture_output=True, text=True, timeout=5)
        for line in res.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 4:
                table[int(parts[0])] = {"ppid": int(parts[1]), "pgid": int(parts[2]), "stat": parts[3],
                                        "command": parts[4] if len(parts) > 4 else ""}
    except Exception:
        pass
    return table


def _is_alive(pid: int, table: Dict[int, Dict[str, Any]]) -> bool:
    """True if pid exists in the table and is not a zombie."""
    info = table.get(pid)
    return bool(info) and not info["stat"].startswith("Z")


def _descendants(pid: int, table: Dict[int, Dict[str, Any]]) -> Set[int]:
    """All transitive children of pid."""
    found: Set[int] = set()
    frontier = [pid]
    while frontier:
        cur = frontier.pop()
        for child, info in table.items():
            if info["ppid"] == cur and child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _group_is_safe(pgid: int, table: Dict[int, Dict[str, Any]]) -> bool:
    """A process group may be signalled as a unit only if it is not ours and every live member looks like an MLX server process."""
    if pgid <= 1 or pgid == os.getpgrp():
        return False
    members = [i for i in table.values() if i["pgid"] == pgid and not i["stat"].startswith("Z")]
    return bool(members) and all(MLX_PROC_SIGNATURE.search(i["command"]) for i in members)


def get_process_rss_mb(pid: int) -> float:
    """Return Resident Set Size in megabytes for pid."""
    try:
        cmd = ["ps", "-p", str(pid), "-o", "rss="]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            kb = int(res.stdout.strip())
            return round(kb / 1024.0, 2)
    except Exception:
        pass
    return 0.0


def _parse_footprint(text: str) -> Optional[float]:
    """Parse `Footprint: 4711 MB` (also KB/GB) from `footprint -p` output into megabytes."""
    m = FOOTPRINT_RE.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = m.group(2).upper()
    return round(value * {"KB": 1 / 1024.0, "MB": 1.0, "GB": 1024.0}[unit], 1)


def get_process_footprint_mb(pid: int) -> float:
    """Physical footprint in MB, including Metal/unified memory that `ps` RSS does not show (0.0 if unavailable)."""
    try:
        res = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        value = _parse_footprint(res.stdout)
        if value is not None:
            return value
    except Exception:
        pass
    return 0.0


def touch_activity() -> None:
    """Update last_activity timestamp for idle watcher."""
    state = load_state()
    now = time.time()
    if state:
        state["last_active"] = now
        save_state(state)


def start_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 model: str = DEFAULT_MODEL, idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
                 wait_ready: bool = True, timeout_sec: int = 45) -> Dict[str, Any]:
    """Start local mlx_lm.server daemon or attach to existing instance."""
    now = time.time()
    existing_pid = get_pid_listening_on_port(port)

    # 1. Check if an active server is already responding on this port
    if existing_pid and is_server_responding(host, port):
        state = load_state()
        state.update({
            "status": "running",
            "pid": existing_pid,
            "host": host,
            "port": port,
            "model": model,
            "attached_existing": True,
            "managed": False,
            "worker_pid": existing_pid,
            "last_active": now,
            "idle_timeout": idle_timeout
        })
        if "started_at" not in state:
            state["started_at"] = now
        save_state(state)
        return {
            "status": "already_running",
            "pid": existing_pid,
            "host": host,
            "port": port,
            "model": model,
            "endpoint": f"http://{host}:{port}/v1",
            "rss_mb": get_process_rss_mb(existing_pid),
            "footprint_mb": get_process_footprint_mb(existing_pid)
        }

    # 2. Port conflict with non-MLX service
    if existing_pid and not is_server_responding(host, port):
        return {
            "status": "error",
            "error": f"Port {port} is occupied by PID {existing_pid} but is not responding as an MLX server."
        }

    # 3. Verify uvx is available
    if not shutil.which("uvx"):
        return {
            "status": "error",
            "error": "uvx binary not found in PATH. Install uv: curl -LsSf https://astral.sh/uv/install.sh"
        }

    # 4. Launch mlx_lm.server as background daemon
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_fp = open(LOG_FILE, "a", encoding="utf-8")
    cmd = [
        "uvx", "--from", "mlx-lm", "mlx_lm.server",
        "--model", model,
        "--port", str(port),
        "--host", host
    ]

    log_fp.write(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting MLX Server ---\n")
    log_fp.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    except Exception as e:
        log_fp.close()
        return {"status": "error", "error": f"Failed to spawn mlx_lm.server: {e}"}

    pid = proc.pid
    state = {
        "status": "starting",
        "pid": pid,
        "host": host,
        "port": port,
        "model": model,
        "started_at": now,
        "last_active": now,
        "idle_timeout": idle_timeout,
        "managed": True,
        "log_file": str(LOG_FILE)
    }
    save_state(state)

    if not wait_ready:
        return {
            "status": "starting",
            "pid": pid,
            "port": port,
            "model": model,
            "endpoint": f"http://{host}:{port}/v1"
        }

    # 5. Wait for readiness gate
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        # Check if process died
        if proc.poll() is not None:
            clear_state()
            return {
                "status": "error",
                "error": f"mlx_lm.server process exited unexpectedly with code {proc.returncode}. See {LOG_FILE}"
            }
        if is_server_responding(host, port):
            worker = get_pid_listening_on_port(port) or pid
            state["status"] = "running"
            state["worker_pid"] = worker
            save_state(state)
            return {
                "status": "started",
                "pid": pid,
                "worker_pid": worker,
                "port": port,
                "model": model,
                "endpoint": f"http://{host}:{port}/v1",
                "boot_latency_sec": round(time.time() - t0, 3),
                "rss_mb": get_process_rss_mb(worker),
                "footprint_mb": get_process_footprint_mb(worker)
            }
        time.sleep(0.5)

    return {
        "status": "timeout",
        "pid": pid,
        "error": f"Server spawned (PID {pid}) but did not respond within {timeout_sec}s. See {LOG_FILE}"
    }


def _terminate(sig: int, workers: List[int], launcher: Optional[int], pgid: Optional[int],
               table: Dict[int, Dict[str, Any]]) -> str:
    """Signal the whole server tree. Uses the process group when it is safe, else the worker/launcher tree."""
    if pgid is not None and _group_is_safe(pgid, table):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        else:
            return "process_group"
    targets: Set[int] = set(workers)
    if launcher:
        targets.add(launcher)
    for pid in list(targets):
        targets |= _descendants(pid, table)
    for pid in targets:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    return "process_tree"


def _survivors(tracked: Set[int], port: int) -> Dict[str, List[int]]:
    """Tracked processes still alive and PIDs still listening on the port."""
    table = _ps_table()
    return {"alive": sorted(p for p in tracked if _is_alive(p, table)), "listeners": get_listening_pids(port)}


def _wait_until_gone(tracked: Set[int], port: int, timeout: float) -> Dict[str, List[int]]:
    """Poll until nothing tracked is alive and nothing listens on the port, or the timeout passes."""
    t0 = time.time()
    left = _survivors(tracked, port)
    while (left["alive"] or left["listeners"]) and time.time() - t0 < timeout:
        time.sleep(0.2)
        left = _survivors(tracked, port)
    return left


def stop_server(port: int = DEFAULT_PORT, force: bool = False) -> Dict[str, Any]:
    """Stop the MLX server tree (launcher and listening worker) and verify that nothing survives.

    The listener on the port is the authority; the PID in the state file (the `uvx` launcher) is only used if it is alive and still
    looks like an MLX process, so a stale or reused PID is never signalled. Returns an error, not "stopped", if anything survives.
    """
    state = load_state()
    table = _ps_table()
    workers = get_listening_pids(port)
    launcher = state.get("pid")
    if launcher and not (_is_alive(launcher, table) and MLX_PROC_SIGNATURE.search(table[launcher]["command"])):
        launcher = None
    if launcher in workers:
        launcher = None
    if not workers and not launcher:
        clear_state()
        return {"status": "not_running", "message": f"No process found listening on port {port}."}

    anchor = workers[0] if workers else launcher
    pgid = table[anchor]["pgid"]
    tracked: Set[int] = set(workers) | ({launcher} if launcher else set())
    for pid in list(tracked):
        tracked |= _descendants(pid, table)
    if _group_is_safe(pgid, table):
        tracked |= {p for p, i in table.items() if i["pgid"] == pgid and not i["stat"].startswith("Z")}
    footprint = max((get_process_footprint_mb(w) for w in workers), default=0.0)

    how = _terminate(signal.SIGTERM, workers, launcher, pgid, table)
    left = _wait_until_gone(tracked, port, 5.0)
    if left["alive"] or left["listeners"]:
        if not force:
            return {"status": "error", "pid": anchor, "survivors": left,
                    "error": "Server did not terminate within 5s. Use --force to kill with SIGKILL."}
        how = _terminate(signal.SIGKILL, workers, launcher, pgid, _ps_table())
        left = _wait_until_gone(tracked, port, 3.0)
        if left["alive"] or left["listeners"]:
            return {"status": "error", "pid": anchor, "survivors": left, "error": "Processes survived SIGKILL."}

    clear_state()
    return {
        "status": "stopped",
        "pid": anchor,
        "worker_pids": workers,
        "launcher_pid": launcher,
        "signalled": how,
        "verified": True,
        "reclaimed_footprint_mb": footprint,
        "reclaimed_rss_mb": footprint,
        "message": f"Stopped the MLX server tree (worker {workers}, launcher {launcher}) and verified nothing is listening on {port}."
    }


def get_status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Dict[str, Any]:
    """Return telemetry of the local MLX server; pid/memory describe the listening worker, not the launcher."""
    state = load_state()
    active = is_server_responding(host, port)
    worker = get_pid_listening_on_port(port)
    state_pid = state.get("pid")
    pid = worker or state_pid
    now = time.time()

    uptime_sec = round(now - state["started_at"], 1) if state.get("started_at") else None
    last_active = state.get("last_active", now)
    idle_sec = round(now - last_active, 1)

    return {
        "is_active": active,
        "pid": pid,
        "launcher_pid": state_pid if state_pid and state_pid != pid else None,
        "host": host,
        "port": port,
        "model": state.get("model", DEFAULT_MODEL),
        "managed": state.get("managed", False),
        "endpoint": f"http://{host}:{port}/v1" if active else None,
        "uptime_sec": uptime_sec,
        "idle_sec": idle_sec if active else None,
        "idle_timeout_sec": state.get("idle_timeout", DEFAULT_IDLE_TIMEOUT),
        "rss_mb": get_process_rss_mb(pid) if pid else 0.0,
        "footprint_mb": get_process_footprint_mb(pid) if pid and active else 0.0
    }


def ensure_ready(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Ensure server is responsive, auto-starting if necessary."""
    touch_activity()
    if is_server_responding(host, port):
        pid = get_pid_listening_on_port(port)
        return {
            "status": "ready",
            "pid": pid,
            "endpoint": f"http://{host}:{port}/v1",
            "already_active": True
        }
    # Auto-start
    return start_server(host=host, port=port, model=model, wait_ready=True)


def reap_if_idle(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 idle_timeout_sec: Optional[int] = None, only_managed: bool = False) -> Dict[str, Any]:
    """Reap server if idle time exceeds configured timeout.

    With only_managed=True a server this manager did not start (an attached or hand-launched daemon) is never reaped; this is the
    mode used by the Protocol 128 boot and closure hooks so a persistent server owned by the human is left alone.
    """
    status = get_status(host, port)
    if not status["is_active"]:
        return {"status": "inactive", "message": "Server is not running; nothing to reap."}
    if only_managed and not status.get("managed"):
        return {"status": "unmanaged", "message": "Server was not started by this manager; leaving it running.",
                "pid": status.get("pid")}

    timeout = idle_timeout_sec or status.get("idle_timeout_sec") or DEFAULT_IDLE_TIMEOUT
    idle = status.get("idle_sec", 0.0)

    if idle >= timeout:
        stop_res = stop_server(port=port, force=True)
        return {
            "status": "reaped",
            "idle_sec": idle,
            "threshold_sec": timeout,
            "reclaimed_rss_mb": stop_res.get("reclaimed_rss_mb", 0.0)
        }

    return {
        "status": "active",
        "idle_sec": idle,
        "threshold_sec": timeout,
        "remaining_sec": round(timeout - idle, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="On-Demand MLX Local Server Lifecycle Controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = subparsers.add_parser("start", help="Start local MLX server daemon")
    p_start.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8080)")
    p_start.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (default: 127.0.0.1)")
    p_start.add_argument("--model", default=DEFAULT_MODEL, help="Model to serve")
    p_start.add_argument("--idle-timeout", type=int, default=DEFAULT_IDLE_TIMEOUT, help="Idle timeout seconds")
    p_start.add_argument("--no-wait", action="store_true", help="Do not wait for server to be responsive")

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop running MLX server daemon")
    p_stop.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port of server to stop")
    p_stop.add_argument("--force", action="store_true", help="Force kill with SIGKILL if SIGTERM times out")

    # status
    p_status = subparsers.add_parser("status", help="Get MLX server status and telemetry")
    p_status.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to probe")
    p_status.add_argument("--host", default=DEFAULT_HOST, help="Host to probe")

    # ensure
    p_ensure = subparsers.add_parser("ensure", help="Ensure server is running (start if needed)")
    p_ensure.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    p_ensure.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    p_ensure.add_argument("--model", default=DEFAULT_MODEL, help="Model to serve")

    # reap
    p_reap = subparsers.add_parser("reap", help="Reap server if idle time exceeded")
    p_reap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to inspect")
    p_reap.add_argument("--timeout", type=int, default=None, help="Override idle timeout in seconds")
    p_reap.add_argument("--only-managed", action="store_true", help="Only reap a server this manager started")

    # touch
    subparsers.add_parser("touch", help="Touch activity timestamp")

    args = parser.parse_args()

    if args.command == "start":
        res = start_server(
            host=args.host,
            port=args.port,
            model=args.model,
            idle_timeout=args.idle_timeout,
            wait_ready=not args.no_wait
        )
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("status") in ("started", "already_running") else 1)

    elif args.command == "stop":
        res = stop_server(port=args.port, force=args.force)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("status") in ("stopped", "not_running") else 1)

    elif args.command == "status":
        res = get_status(host=args.host, port=args.port)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    elif args.command == "ensure":
        res = ensure_ready(host=args.host, port=args.port, model=args.model)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("status") in ("ready", "started", "already_running") else 1)

    elif args.command == "reap":
        res = reap_if_idle(port=args.port, idle_timeout_sec=args.timeout, only_managed=args.only_managed)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    elif args.command == "touch":
        touch_activity()
        print(json.dumps({"status": "touched", "time": time.time()}))
        sys.exit(0)


if __name__ == "__main__":
    main()
