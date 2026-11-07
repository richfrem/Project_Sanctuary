"""Where `checkin` stores things: board/leases are shared via the main checkout; the CHECKIN event is the running checkout's own file.

A check-in made from a linked worktree is therefore an uncommitted edit in that worktree (it can be stranded if the worktree is
discarded). `checkin` must report the exact path, and it must never export or capture preservation data itself.
"""
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import manage_coordination as mc  # noqa: E402


def git(root, *a):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=root, capture_output=True, text=True, check=True).stdout


def make_main_and_worktree(tmp_path):
    main = tmp_path / "main"; main.mkdir()
    git(main, "init", "-q", "-b", "main")
    (main / ".agent/learning").mkdir(parents=True)
    (main / ".agent/learning/events.jsonl").write_text("")
    (main / "README.md").write_text("x\n")
    git(main, "add", "-A"); git(main, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    git(main, "worktree", "add", "-q", "-b", "feat/x", str(wt))
    return main, wt


def test_worktree_checkin_event_lands_in_the_worktree_not_main(tmp_path):
    main, wt = make_main_and_worktree(tmp_path)
    res = mc.checkin(wt, "task-1", "agent-a", handoff="HANDOFF-STORAGE-TEST", no_usage=True)
    wt_events = wt / ".agent/learning/events.jsonl"
    assert res["events_file"] == str(wt_events)                              # reported, so nobody has to guess
    assert "HANDOFF-STORAGE-TEST" in wt_events.read_text()
    assert (main / ".agent/learning/events.jsonl").read_text() == ""         # main's copy is NOT touched
    assert "events.jsonl" in git(wt, "status", "--short")                    # uncommitted: must be committed with the branch
    # The untracked board/leases are the shared, main-checkout copy.
    assert (main / ".agent/coordination/leases.jsonl").exists()
    assert not (wt / ".agent/coordination/leases.jsonl").exists()


def test_checkin_appends_exactly_one_event_and_never_touches_preservation_files(tmp_path):
    main, wt = make_main_and_worktree(tmp_path)
    for p in ("huggingface/data/soul_traces.jsonl", "huggingface/metadata/coverage_manifest.json"):
        f = wt / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_text("UNCHANGED\n")
    mc.checkin(wt, "task-2", "agent-a", handoff="one", no_usage=True)
    lines = [json.loads(x) for x in (wt / ".agent/learning/events.jsonl").read_text().splitlines()]
    assert [e["mission_id"] for e in lines] == ["task-2"] and lines[0]["event_type"] == "CHECKIN"
    for p in ("huggingface/data/soul_traces.jsonl", "huggingface/metadata/coverage_manifest.json"):
        assert (wt / p).read_text() == "UNCHANGED\n"                         # check-in does not export
    assert not (wt / ".agent/coordination/soul_preservation.json").exists()
