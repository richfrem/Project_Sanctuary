#!/usr/bin/env python3
"""test_coordination.py - Pytest suite for multi-agent concurrency flight board (MISSION-026).
Purpose: Verifies checkout, checkin, target/resource conflict detection, heartbeat, and flight board rendering.
Usage:
    python3 -m pytest plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_coordination.py -v
"""

import sys
import json
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import manage_coordination as mc


@pytest.fixture
def temp_project_root(tmp_path: Path) -> Path:
    """Prepare a temporary project structure with missions and coordination dirs."""
    missions_dir = tmp_path / "LEARNING" / "research" / "missions"
    missions_dir.mkdir(parents=True, exist_ok=True)
    sample_mission = {
        "id": "MISSION-026",
        "title": "Sample Concurrency Mission",
        "status": "PROPOSED",
        "priority_score": 200.0
    }
    (missions_dir / "MISSION-026.json").write_text(json.dumps(sample_mission, indent=2))
    return tmp_path


def test_checkout_creates_lease_and_updates_board(temp_project_root: Path):
    res = mc.checkout(
        temp_project_root,
        mission="MISSION-026",
        agent="antigravity",
        targets=["plugins/sanctuary-core/scripts/forge_llm.py"],
        resources=["mlx-server:8080"],
        worktree=".worktrees/agy-mission-026",
        ttl_minutes=60
    )
    assert res["status"] == "success"
    assert res["mission"] == "MISSION-026"
    assert res["agent"] == "antigravity"

    board = mc.load_board(temp_project_root)
    assert len(board["leases"]) == 1
    lease = board["leases"][0]
    assert lease["agent_id"] == "antigravity"
    assert lease["mission_id"] == "MISSION-026"
    assert "mlx-server:8080" in lease["exclusive_resources"]

    # Verify mission manifest was updated to IN_PROGRESS
    mission_file = temp_project_root / "LEARNING" / "research" / "missions" / "MISSION-026.json"
    data = json.loads(mission_file.read_text())
    assert data["status"] == "IN_PROGRESS"
    assert data["assignee"] == "antigravity"


def test_conflict_detection_blocks_overlapping_target(temp_project_root: Path):
    # Agent 1 checks out forge_llm.py
    mc.checkout(
        temp_project_root,
        mission="MISSION-022",
        agent="claude-code",
        targets=["plugins/sanctuary-core/scripts/forge_llm.py"],
        worktree=".worktrees/claude-mission-022"
    )

    # Agent 2 attempts to checkout overlapping forge_llm.py
    res = mc.checkout(
        temp_project_root,
        mission="MISSION-027",
        agent="codex",
        targets=["plugins/sanctuary-core/scripts/forge_llm.py"],
        worktree=".worktrees/codex-mission-027"
    )
    assert res["status"] == "conflict"
    assert len(res["conflicts"]) == 1
    assert res["conflicts"][0]["holding_agent"] == "claude-code"


def test_conflict_detection_blocks_exclusive_resource(temp_project_root: Path):
    # Agent 1 locks mlx-server:8080
    mc.checkout(
        temp_project_root,
        mission="MISSION-022",
        agent="claude-code",
        resources=["mlx-server:8080"]
    )

    # Agent 2 attempts to lock mlx-server:8080 on a different task
    res = mc.checkout(
        temp_project_root,
        mission="MISSION-023",
        agent="antigravity",
        resources=["mlx-server:8080"]
    )
    assert res["status"] == "conflict"
    assert "mlx-server:8080" in res["conflicts"][0]["conflicting_resources"]


def test_force_checkout_bypasses_conflict(temp_project_root: Path):
    mc.checkout(
        temp_project_root,
        mission="MISSION-022",
        agent="claude-code",
        resources=["mlx-server:8080"]
    )
    res = mc.checkout(
        temp_project_root,
        mission="MISSION-023",
        agent="antigravity",
        resources=["mlx-server:8080"],
        force=True
    )
    assert res["status"] == "success"


def test_heartbeat_extends_lease(temp_project_root: Path):
    mc.checkout(
        temp_project_root,
        mission="MISSION-026",
        agent="antigravity",
        ttl_minutes=10,
        allow_unlocked=True  # this test only exercises the heartbeat; a lease with no locks must now be requested explicitly
    )
    res = mc.heartbeat(temp_project_root, mission="MISSION-026", agent="antigravity", extend_minutes=45)
    assert res["status"] == "success"

    board = mc.load_board(temp_project_root)
    lease = board["leases"][0]
    assert lease["expires_at"] > time.time() + (40 * 60)


def test_checkin_releases_lease(temp_project_root: Path):
    mc.checkout(
        temp_project_root,
        mission="MISSION-026",
        agent="antigravity",
        targets=["foo.py"]
    )
    assert len(mc.load_board(temp_project_root)["leases"]) == 1

    checkin_res = mc.checkin(temp_project_root, mission="MISSION-026", agent="antigravity", status="COMPLETED", pr=278)
    assert checkin_res["status"] == "success"
    assert len(mc.load_board(temp_project_root)["leases"]) == 0

    # Verify event audit log
    _, log_file = mc.get_coordination_paths(temp_project_root)
    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "CHECKOUT"
    assert json.loads(lines[1])["event_type"] == "CHECKIN"


def test_board_rendering_and_stale_detection(temp_project_root: Path):
    # Add an active lease and an expired lease directly
    now = time.time()
    board = {
        "leases": [
            {
                "agent_id": "claude-code",
                "mission_id": "MISSION-022",
                "status": "EXECUTING",
                "worktree": ".worktrees/claude-mission-022",
                "locked_targets": ["forge_llm.py"],
                "exclusive_resources": ["mlx:8080"],
                "checked_out_timestamp": now - 300,
                "last_heartbeat": now - 30,
                "expires_at": now + 1800
            },
            {
                "agent_id": "zombie-agent",
                "mission_id": "MISSION-999",
                "status": "EXECUTING",
                "worktree": ".worktrees/zombie",
                "locked_targets": ["stale.py"],
                "exclusive_resources": [],
                "checked_out_timestamp": now - 7200,
                "last_heartbeat": now - 7000,
                "expires_at": now - 100  # Expired
            }
        ]
    }
    mc.save_board(temp_project_root, board)

    active = mc.get_active_board(temp_project_root)
    assert len(active) == 2
    assert active[0]["is_stale"] is False
    assert active[1]["is_stale"] is True
    assert active[1]["status"] == "STALE"

    text = mc.render_board_text(active)
    assert "[EXECUTING]" in text
    assert "[STALE]" in text
    assert "claude-code" in text
    assert "zombie-agent" in text

    # Clean stale
    clean_res = mc.clean_stale(temp_project_root)
    assert clean_res["removed_count"] == 1
    assert len(mc.load_board(temp_project_root)["leases"]) == 1
