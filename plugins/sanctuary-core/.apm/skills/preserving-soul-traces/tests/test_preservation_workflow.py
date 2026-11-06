"""Bounded lifecycle integration fixtures; no research, model calls or network."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preservation_workflow as W
import soul_coverage as S
from test_soul_coverage import make_repo, git, alltext, text_of

SKILLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILLS / "managing-sovereign-memory/scripts"))
from manage_backlog import BacklogManager
sys.path.insert(0, str(SKILLS / "operating-sanctuary-guardian/scripts"))
import merge_prep


def publish(root):
    git(root, "add", "-A"); git(root, "commit", "-qm", "publish sources"); git(root, "push", "-q")


def resolve(root, mid="MISSION-901"):
    return BacklogManager(root).record_result(mid, "FALSIFIED", "FINAL-EVIDENCE", lessons=["FINAL-LESSON"])


def test_resolution_obligation_is_atomic_with_honest_verdict_and_survives_restart():
    root = make_repo()
    m = resolve(root)
    assert m["preservation_obligation"]["required"]
    assert json.loads((root / "LEARNING/missions/MISSION-901.json").read_text())["verdict"] == "FALSIFIED"
    # No progress file is required for discovery after a crash.
    state = W.bootstrap(root)
    assert state["status"] == "pending_publication"
    assert state["missions"]["MISSION-901"]["verdict"] == "FALSIFIED"
    assert state["resume"]
    publish(root)
    assert W.bootstrap(root)["status"] == "local_complete"
    content = alltext(root)
    for needle in ("RESOLVED", "FALSIFIED", "FINAL-EVIDENCE", "FINAL-LESSON", "FINDING-901", "LIMITATION-901", "run.py", "results.json"):
        assert needle in content


def test_post_integration_supported_path_invokes_capture_and_keeps_science_separate():
    root = make_repo(); resolve(root); publish(root)
    report = merge_prep.regenerate(root)
    # compile may normalize routing; publish first if it changed a source.
    if report["preservation"]["status"] == "pending_publication":
        publish(root); report = merge_prep.regenerate(root)
    assert report["preservation"]["status"] == "local_complete"
    assert "FALSIFIED" in text_of(root, "mission__MISSION-901")
    assert "pending authorization" in report["preservation"]["remote"]
    assert json.loads((root / "LEARNING/missions/MISSION-901.json").read_text())["verdict"] == "FALSIFIED"


def test_unpublished_artifacts_and_dirty_bytes_wait_then_resume():
    root = make_repo(publish=False)
    assert W.run(root)["status"] == "pending_publication"
    git(root, "push", "-qu", "origin", "main")
    artifact = root / "LEARNING/experiments/mission-901/run.py"
    artifact.write_text("print('revised')\n")
    assert W.run(root)["status"] == "pending_publication"
    git(root, "add", "-A"); git(root, "commit", "-qm", "artifact revision")
    assert W.run(root)["status"] == "pending_publication"
    git(root, "push", "-q")
    assert W.bootstrap(root)["status"] == "local_complete"
    assert S.sha(artifact.read_bytes()) in alltext(root)


def test_failed_export_visible_retryable_and_does_not_rewrite_verdict():
    root = make_repo(); resolve(root); publish(root)
    original = (root / "LEARNING/missions/MISSION-901.json").read_bytes()
    real_export = S.export
    def broken(*args, **kwargs):
        if not kwargs.get("dry"):
            raise OSError("fixture failure")
        return real_export(*args, **kwargs)
    with patch.object(S, "export", broken):
        assert W.run(root)["status"] == "failed"
    assert json.loads((root / W.STATE).read_text())["status"] == "failed"
    assert (root / "LEARNING/missions/MISSION-901.json").read_bytes() == original
    assert W.bootstrap(root)["status"] == "local_complete"


def test_unchanged_closure_and_receipts_do_not_reexport_or_invalidate_hashes():
    root = make_repo(); assert W.run(root)["status"] == "local_complete"
    first = [(root / p).read_bytes() for p in (S.DATASET, S.MANIFEST)]
    receipt = {"verified": True, "revision": "fixture", "time": "now",
               "local_dataset_sha256": S.sha(first[0]), "local_manifest_sha256": S.sha(first[1])}
    (root / S.RECEIPTS).write_text(json.dumps(receipt)+"\n")
    publish(root)
    for _ in range(2):
        state = W.run(root)
        assert state["status"] == "local_complete"
        assert state["facts"]["remote"].startswith("verified")
        assert [(root / p).read_bytes() for p in (S.DATASET, S.MANIFEST)] == first
    receipt["local_manifest_sha256"] = "wrong"
    (root / S.RECEIPTS).write_text(json.dumps(receipt)+"\n")
    assert S.check(root)["remote"].startswith("NOT VERIFIED")


def test_interruption_pending_is_discoverable_and_retried():
    root = make_repo()
    # SIGKILL releases the OS lock and leaves a durable intent record.
    script = """import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]); import preservation_workflow as W
root=Path(sys.argv[2])
with W.preservation_lock(root):
 W.atomic_json(root/W.STATE, {'status':'pending'})
 os._exit(17)
"""
    r = subprocess.run([sys.executable, "-c", script, str(W.HERE), str(root)])
    assert r.returncode == 17
    assert W.bootstrap(root)["status"] == "local_complete"


def test_two_simultaneous_completions_and_exports_cannot_lose_updates():
    root = make_repo()
    m = json.loads((root / "LEARNING/missions/MISSION-901.json").read_text()); m["id"] = "MISSION-902"
    (root / "LEARNING/missions/MISSION-902.json").write_text(json.dumps(m)); publish(root)
    script = """import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]); from manage_backlog import BacklogManager
BacklogManager(Path(sys.argv[2])).record_result(sys.argv[3], 'FALSIFIED', sys.argv[3]+'-evidence')
"""
    workers = [subprocess.Popen([sys.executable, "-c", script, str(SKILLS / "managing-sovereign-memory/scripts"), str(root), mid], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for mid in ("MISSION-901", "MISSION-902")]
    for worker in workers:
        out,err = worker.communicate(timeout=30); assert worker.returncode == 0, err
    publish(root)
    workers = [subprocess.Popen([sys.executable, str(W.HERE / "preservation_workflow.py"), "--repo-root", str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    for worker in workers:
        out,err = worker.communicate(timeout=30); assert worker.returncode == 0, (out,err)
    for mid in ("MISSION-901", "MISSION-902"):
        assert mid+"-evidence" in text_of(root, "mission__"+mid)
        assert mid in json.loads((root / W.STATE).read_text())["missions"]
    records = S.load_dataset(root)
    assert S.check(root)["local_export"] == "current"
    assert len(records) == len({r["id"] for r in records.values()})
