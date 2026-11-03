#!/usr/bin/env python3
"""mlx_server_manager.py - On-Demand MLX Local Server Lifecycle Controller.
Purpose: Manages background mlx_lm.server daemon lifecycle, health gates, and idle memory reclamation.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 Execution & Resource Management).
Key Inputs: Server host/port, MLX model repository, idle timeout (seconds), action command.
Key Outputs: Process PID, operational status, memory telemetry (RSS MB), and JSON lifecycle events.
Usage:
    python3 mlx_server_manager.py start [--port 8080] [--idle-timeout 900]
    python3 mlx_server_manager.py stop [--force]
    python3 mlx_server_manager.py status
    python3 mlx_server_manager.py ensure
    python3 mlx_server_manager.py reap
Key Functions: start_server(), stop_server(), get_status(), ensure_ready(), reap_if_idle(), touch_activity().
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
from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_HOST = os.environ.get("MLX_SERVER_HOST_IP", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MLX_SERVER_PORT", "8080"))
DEFAULT_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
DEFAULT_IDLE_TIMEOUT = int(os.environ.get("MLX_IDLE_TIMEOUT_SEC", "900"))  # 15 minutes

STATE_DIR = Path(".agent/runtime")
STATE_FILE = STATE_DIR / "mlx_server.json"
FALLBACK_STATE_FILE = Path("/tmp/sanctuary_mlx_server.json")
LOG_FILE = STATE_DIR / "mlx_server.log"


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


def get_pid_listening_on_port(port: int = DEFAULT_PORT) -> Optional[int]:
    """Find process ID currently listening on port."""
    try:
        cmd = ["lsof", "-ti", f":{port}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            pids = res.stdout.strip().split()
            return int(pids[0])
    except Exception:
        pass
    return None


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
            "rss_mb": get_process_rss_mb(existing_pid)
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
            state["status"] = "running"
            save_state(state)
            return {
                "status": "started",
                "pid": pid,
                "port": port,
                "model": model,
                "endpoint": f"http://{host}:{port}/v1",
                "boot_latency_sec": round(time.time() - t0, 3),
                "rss_mb": get_process_rss_mb(pid)
            }
        time.sleep(0.5)

    return {
        "status": "timeout",
        "pid": pid,
        "error": f"Server spawned (PID {pid}) but did not respond within {timeout_sec}s. See {LOG_FILE}"
    }


def stop_server(port: int = DEFAULT_PORT, force: bool = False) -> Dict[str, Any]:
    """Gracefully stop mlx_lm.server and release Metal unified memory buffer."""
    state = load_state()
    pid = state.get("pid") or get_pid_listening_on_port(port)

    if not pid:
        clear_state()
        return {"status": "not_running", "message": f"No process found listening on port {port}."}

    initial_rss = get_process_rss_mb(pid)

    # 1. Send SIGTERM for graceful shutdown
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        clear_state()
        return {"status": "not_running", "message": f"Process PID {pid} was already terminated."}
    except Exception as e:
        return {"status": "error", "error": f"Failed to signal PID {pid}: {e}"}

    # 2. Wait up to 5 seconds for termination
    t0 = time.time()
    terminated = False
    while time.time() - t0 < 5.0:
        if get_pid_listening_on_port(port) != pid:
            terminated = True
            break
        time.sleep(0.2)

    # 3. Force kill if requested and still running
    if not terminated:
        if force:
            try:
                os.kill(pid, signal.SIGKILL)
                terminated = True
            except Exception:
                pass
        else:
            return {
                "status": "error",
                "pid": pid,
                "error": "Process did not terminate within 5s. Use --force to kill with SIGKILL."
            }

    clear_state()
    return {
        "status": "stopped",
        "pid": pid,
        "reclaimed_rss_mb": initial_rss,
        "message": f"Successfully stopped mlx_lm.server (PID {pid}) and released memory."
    }


def get_status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Dict[str, Any]:
    """Return comprehensive telemetry of local MLX server."""
    state = load_state()
    active = is_server_responding(host, port)
    pid = state.get("pid") or get_pid_listening_on_port(port)
    now = time.time()

    uptime_sec = round(now - state["started_at"], 1) if state.get("started_at") else None
    last_active = state.get("last_active", now)
    idle_sec = round(now - last_active, 1)

    return {
        "is_active": active,
        "pid": pid,
        "host": host,
        "port": port,
        "model": state.get("model", DEFAULT_MODEL),
        "endpoint": f"http://{host}:{port}/v1" if active else None,
        "uptime_sec": uptime_sec,
        "idle_sec": idle_sec if active else None,
        "idle_timeout_sec": state.get("idle_timeout", DEFAULT_IDLE_TIMEOUT),
        "rss_mb": get_process_rss_mb(pid) if pid else 0.0
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
                 idle_timeout_sec: Optional[int] = None) -> Dict[str, Any]:
    """Reap server if idle time exceeds configured timeout."""
    status = get_status(host, port)
    if not status["is_active"]:
        return {"status": "inactive", "message": "Server is not running; nothing to reap."}

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
        res = reap_if_idle(port=args.port, idle_timeout_sec=args.timeout)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    elif args.command == "touch":
        touch_activity()
        print(json.dumps({"status": "touched", "time": time.time()}))
        sys.exit(0)


if __name__ == "__main__":
    main()
