"""Flight-board files are untracked runtime state: one shared board across worktrees, and handoff notes live in tracked events.jsonl."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import manage_coordination as mc  # noqa: E402
import relay  # noqa: E402
import merge_prep  # noqa: E402

REPO = HERE.parents[6]


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=str(cwd), capture_output=True, text=True, check=True)


def _repo_with_worktree():
    main = Path(tempfile.mkdtemp()).resolve()
    _git(main, "init", "-q", "-b", "main")
    _git(main, "config", "user.email", "t@t"); _git(main, "config", "user.name", "t")
    (main / "f.txt").write_text("x"); _git(main, "add", "-A"); _git(main, "commit", "-qm", "init")
    wt = main / ".worktrees" / "wt1"
    _git(main, "worktree", "add", "-q", str(wt), "-b", "feat/wt1")
    return main, wt


def test_worktree_and_main_share_one_board():
    main, wt = _repo_with_worktree()
    b_main, l_main = mc.get_coordination_paths(main)
    b_wt, l_wt = mc.get_coordination_paths(wt)
    assert b_main.resolve() == b_wt.resolve() and l_main.resolve() == l_wt.resolve()
    assert str(b_wt.resolve()).startswith(str(main))
    assert not (wt / ".agent" / "coordination").exists()  # nothing written into the worktree copy


def test_lease_taken_in_a_worktree_is_visible_from_main():
    main, wt = _repo_with_worktree()
    mc.save_board(wt, {"leases": [{"agent_id": "a", "mission_id": "M-1"}]})
    assert mc.load_board(main)["leases"][0]["mission_id"] == "M-1"


def test_plain_directory_without_git_keeps_its_own_paths():
    d = Path(tempfile.mkdtemp()).resolve()
    b, _ = mc.get_coordination_paths(d)
    assert b.resolve() == (d / ".agent/coordination/board.json").resolve()


def test_checkin_mirrors_handoff_to_tracked_events_without_the_bulky_usage_block():
    root = Path(tempfile.mkdtemp()).resolve()
    mc.save_board(root, {"leases": [{"agent_id": "claude-code", "mission_id": "M-9", "locked_targets": [], "expires_at": 9e12}]})
    res = mc.checkin(root, mission="M-9", agent="claude-code", status="COMPLETED", pr=7, handoff="did X; do not redo Y; next Z", no_usage=True)
    assert res["status"] == "success"
    ev = [json.loads(l) for l in (root / ".agent/learning/events.jsonl").read_text().splitlines() if l.strip()]
    chk = [e for e in ev if e.get("event_type") == "CHECKIN"]
    assert len(chk) == 1 and chk[0]["handoff"].startswith("did X") and chk[0]["pr_number"] == 7
    assert "usage" not in chk[0]


def test_relay_reads_handoffs_from_events_when_leases_file_is_absent():
    root = Path(tempfile.mkdtemp()).resolve()
    (root / ".agent/learning").mkdir(parents=True)
    (root / ".agent/learning/events.jsonl").write_text(json.dumps({
        "event_type": "CHECKIN", "agent_id": "codex", "mission_id": "M-5", "outcome_status": "COMPLETED", "pr_number": 3,
        "timestamp": "2026-11-05T01:00:00+00:00", "handoff": "left a note"}) + "\n")
    h = relay.build_relay(root)["handovers"]
    assert h and h[0]["mission"] == "M-5" and h[0]["handoff"] == "left a note"


def test_relay_does_not_double_count_a_checkin_present_in_both_files():
    root = Path(tempfile.mkdtemp()).resolve()
    (root / ".agent/learning").mkdir(parents=True); (root / ".agent/coordination").mkdir(parents=True)
    e = {"event_type": "CHECKIN", "agent_id": "codex", "mission_id": "M-5", "outcome_status": "COMPLETED", "pr_number": 3,
         "timestamp": "2026-11-05T01:00:00+00:00", "handoff": "left a note"}
    (root / ".agent/learning/events.jsonl").write_text(json.dumps(e) + "\n")
    (root / ".agent/coordination/leases.jsonl").write_text(json.dumps(e) + "\n")
    assert len(relay.build_relay(root)["handovers"]) == 1


def test_repo_no_longer_tracks_the_flight_board_files_and_ignores_them():
    tracked = _git(REPO, "ls-files", ".agent/coordination").stdout.split()
    assert tracked == [], tracked
    ign = subprocess.run(["git", "check-ignore", ".agent/coordination/board.json", ".agent/coordination/leases.jsonl"],
                         cwd=str(REPO), capture_output=True, text=True)
    assert ign.returncode == 0 and ".agent/coordination/board.json" in ign.stdout and ".agent/coordination/leases.jsonl" in ign.stdout


def test_only_events_log_needs_the_union_merge_driver_now():
    assert merge_prep.UNION_LOGS == [".agent/learning/events.jsonl"]
    assert merge_prep.missing_union_rules(REPO) == []


def test_verify_impact_resolves_when_called_from_inside_worktree():
    main, wt = _repo_with_worktree()
    (main / "LEARNING" / "missions").mkdir(parents=True)
    m_json = main / "LEARNING" / "missions" / "MISSION-999.json"
    m_json.write_text(json.dumps({"id": "MISSION-999", "projected_impact": {"paths": ["f.txt"]}}))
    _git(main, "add", "-A"); _git(main, "commit", "-qm", "add mission manifest")
    _git(wt, "merge", "main", "-qm", "sync main")
    mc.save_board(main, {"leases": [{"agent_id": "test", "mission_id": "MISSION-999", "worktree": ".worktrees/wt1", "locked_targets": ["f.txt"]}]})
    # Invoking verify_impact with root=wt (as when run from inside worktree) must resolve wt cleanly and not double-nest
    res = mc.verify_impact(wt, "MISSION-999")
    assert res.get("status") == "success"

