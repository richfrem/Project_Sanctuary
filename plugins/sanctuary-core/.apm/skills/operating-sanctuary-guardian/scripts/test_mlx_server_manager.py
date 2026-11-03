#!/usr/bin/env python3
"""test_mlx_server_manager.py - Test Suite for On-Demand MLX Server Lifecycle Controller.
Purpose: Comprehensive unit & integration tests for MLX server lifecycle, idle detection, and gates.
Layer: sanctuary-core / operating-sanctuary-guardian (Test Automation & Verification).
Key Inputs: Local server manager methods, state files, mock process states.
Key Outputs: Pytest assertions verifying deterministic lifecycle state transitions.
Usage:
    pytest test_mlx_server_manager.py -v
Key Functions: test_state_persistence(), test_touch_activity(), test_idle_reap_logic(), test_status_contract().
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path

# Ensure script directory is in sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mlx_server_manager as msm


@pytest.fixture(autouse=True)
def isolated_runtime_state(tmp_path, monkeypatch):
    """Isolate state file directory to temporary path."""
    test_state_file = tmp_path / "test_mlx_server.json"
    monkeypatch.setattr(msm, "_get_active_state_path", lambda: test_state_file)
    yield


def test_state_persistence():
    """Verify runtime state can be written, read, and cleared."""
    state_data = {
        "status": "running",
        "pid": 99999,
        "host": "127.0.0.1",
        "port": 8080,
        "model": "test-model",
        "started_at": 1000.0,
        "last_active": 1000.0,
        "idle_timeout": 900
    }
    msm.save_state(state_data)
    loaded = msm.load_state()
    assert loaded == state_data

    msm.clear_state()
    assert msm.load_state() == {}


def test_touch_activity():
    """Verify touch_activity updates the last_active timestamp."""
    initial_time = 1000.0
    msm.save_state({"status": "running", "last_active": initial_time})

    time.sleep(0.01)
    msm.touch_activity()
    updated = msm.load_state()
    assert updated["last_active"] > initial_time


def test_reap_if_idle_not_exceeded(monkeypatch):
    """Verify server is NOT reaped when idle time is below threshold."""
    now = time.time()
    msm.save_state({
        "status": "running",
        "pid": 12345,
        "port": 8080,
        "started_at": now - 100,
        "last_active": now - 30,  # 30 seconds ago
        "idle_timeout": 900
    })
    monkeypatch.setattr(msm, "is_server_responding", lambda h, p: True)

    res = msm.reap_if_idle(idle_timeout_sec=900)
    assert res["status"] == "active"
    assert res["idle_sec"] >= 29
    assert res["remaining_sec"] > 0


def test_reap_if_idle_exceeded(monkeypatch):
    """Verify server IS reaped when idle time exceeds threshold."""
    now = time.time()
    msm.save_state({
        "status": "running",
        "pid": 12345,
        "port": 8080,
        "started_at": now - 2000,
        "last_active": now - 1000,  # 1000 seconds ago
        "idle_timeout": 900
    })
    monkeypatch.setattr(msm, "is_server_responding", lambda h, p: True)

    stopped_called = []
    monkeypatch.setattr(msm, "stop_server", lambda port, force: stopped_called.append((port, force)) or {"status": "stopped", "reclaimed_rss_mb": 4430.0})

    res = msm.reap_if_idle(idle_timeout_sec=900)
    assert res["status"] == "reaped"
    assert len(stopped_called) == 1
    assert stopped_called[0] == (8080, True)
    assert res["reclaimed_rss_mb"] == 4430.0


def test_status_contract(monkeypatch):
    """Verify get_status returns complete telemetry dictionary."""
    monkeypatch.setattr(msm, "is_server_responding", lambda h, p: True)
    monkeypatch.setattr(msm, "get_pid_listening_on_port", lambda p: 54321)
    monkeypatch.setattr(msm, "get_process_rss_mb", lambda p: 4431.5)

    now = time.time()
    msm.save_state({
        "status": "running",
        "pid": 54321,
        "port": 8080,
        "started_at": now - 120,
        "last_active": now - 10,
        "idle_timeout": 900,
        "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    })

    st = msm.get_status()
    assert st["is_active"] is True
    assert st["pid"] == 54321
    assert st["port"] == 8080
    assert st["model"] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    assert st["endpoint"] == "http://127.0.0.1:8080/v1"
    assert st["rss_mb"] == 4431.5
    assert st["idle_timeout_sec"] == 900


def test_start_server_already_running(monkeypatch):
    """Verify start_server detects existing running instance without collision."""
    monkeypatch.setattr(msm, "is_server_responding", lambda h, p: True)
    monkeypatch.setattr(msm, "get_pid_listening_on_port", lambda p: 74300)
    monkeypatch.setattr(msm, "get_process_rss_mb", lambda p: 25.0)

    res = msm.start_server(port=8080)
    assert res["status"] == "already_running"
    assert res["pid"] == 74300
    assert res["port"] == 8080
    assert "endpoint" in res


def test_start_server_port_conflict(monkeypatch):
    """Verify start_server detects non-MLX port conflict and fails safely."""
    monkeypatch.setattr(msm, "is_server_responding", lambda h, p: False)
    monkeypatch.setattr(msm, "get_pid_listening_on_port", lambda p: 88888)

    res = msm.start_server(port=8080)
    assert res["status"] == "error"
    assert "occupied" in res["error"]
