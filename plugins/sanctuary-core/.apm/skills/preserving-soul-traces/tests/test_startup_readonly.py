"""Startup must only REPORT preservation status; capture/recovery belong to closure.

Hermetic fixtures: a temp git repo with a bare origin, the real hook and Guardian code pointed at it, and the
machine-touching probes (MLX server reap, localhost probe) stubbed. Nothing here touches the developer's checkout.
"""
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preservation_workflow as W
import soul_coverage as S
from test_soul_coverage import EVENTS, git, make_repo, text_of

SKILLS = Path(__file__).resolve().parents[2]
HOOK = SKILLS.parent / "hooks" / "session_start_wakeup.py"
NEW_HANDOFF = "FINAL-CHECKIN-HANDOFF-777 export must include this"


def publish(root, msg="publish sources"):
    git(root, "add", "-A"); git(root, "commit", "-qm", msg); git(root, "push", "-q")


def watched(root):
    """Every preservation file startup must leave byte-identical (None = must stay absent)."""
    names = (S.DATASET, S.MANIFEST, Path("huggingface/metadata/manifest.json"), W.STATE, S.RECEIPTS)
    return {str(n): ((root / n).read_bytes() if (root / n).exists() else None) for n in names}


def with_guardian_scripts(root):
    """Give the fixture the real Guardian + preservation scripts so the hook resolves them for project_root=root."""
    for skill, parts in (("operating-sanctuary-guardian", ("scripts",)), ("preserving-soul-traces", ("scripts", "references"))):
        for part in parts:                                    # references/ holds coverage-policy.json
            shutil.copytree(SKILLS / skill / part, root / "plugins/sanctuary-core/.apm/skills" / skill / part,
                            ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    return root


def stale_repo():
    """Exported + published, then a check-in event committed and published AFTER the export (the evidenced gap)."""
    root = make_repo()
    assert W.run(root)["status"] == "local_complete"
    publish(root, "export")
    with (root / ".agent/learning/events.jsonl").open("a") as f:
        f.write(json.dumps({"event_type": "CHECKIN", "mission_id": "late", "agent_id": "x", "handoff": NEW_HANDOFF,
                            "timestamp": "2026-01-02T00:00:00Z"}) + "\n")
    publish(root, "late check-in")
    return root


def run_startup_chain(root):
    """The real SessionStart path: hook -> guardian_wakeup.check_layer1_health -> check_soul_preservation -> bootstrap."""
    with_guardian_scripts(root)
    spec = importlib.util.spec_from_file_location("hook_under_test", HOOK)
    hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
    fake_msm = types.SimpleNamespace(reap_if_idle=lambda **k: {"status": "none"})
    boom = AssertionError("startup attempted a preservation write")
    with patch.dict(sys.modules, {"mlx_server_manager": fake_msm}), \
         patch("urllib.request.urlopen", side_effect=OSError("no network in tests")), \
         patch.object(W, "run", side_effect=boom), patch.object(S, "export", side_effect=boom), \
         patch.object(S, "write_dataset", side_effect=boom), patch.object(W, "atomic_json", side_effect=boom):
        return hook.execute_bootloader_in_process(root)


def soul_line(digest):
    return next(line for line in digest.splitlines() if "soul-preservation" in line)


def test_stale_sources_are_reported_and_nothing_is_written_by_startup():
    root = stale_repo()
    assert S.quick_status(root)[0] == "WARN"
    before = watched(root)
    code, digest, _ = run_startup_chain(root)
    assert code == 0
    line = soul_line(digest)
    assert "[WARN]" in line and "STALE" in line and "local preservation stale" in line
    assert "soul_coverage.py closure" in line                       # pending work is visible, not hidden
    assert watched(root) == before                                  # dataset, manifests, state, receipts untouched
    assert NEW_HANDOFF not in (root / S.DATASET).read_text()


@pytest.mark.parametrize("recorded", ["pending", "failed", "pending_publication"])
def test_recorded_obligations_survive_startup_unchanged(recorded):
    root = stale_repo()
    state = {"status": recorded, "missions": {"MISSION-901": {"manifest_sha256": "x", "verdict": "FALSIFIED"}},
             "problems": ["fixture problem"], "resume": W.RESUME, "remote": "pending"}
    W.atomic_json(root / W.STATE, state)
    before = watched(root)
    assert W.bootstrap(root)["status"] == recorded                   # reported exactly as recorded
    code, digest, _ = run_startup_chain(root)
    assert code == 0 and recorded in soul_line(digest)
    assert watched(root) == before
    assert json.loads((root / W.STATE).read_text()) == state        # a later closure still sees the obligation


def test_closure_captures_the_committed_checkin_event():
    root = stale_repo()
    assert W.bootstrap(root)["status"] == "stale"
    assert NEW_HANDOFF not in text_of(root, "events__CHECKIN__001")
    state = W.run(root)                                              # the closure path
    assert state["status"] == "local_complete"
    assert NEW_HANDOFF in text_of(root, "events__CHECKIN__001")
    assert S.check(root)["local_export"] == "current"
    assert "STALE" not in S.quick_status(root)[1]


def test_repeated_closure_has_no_content_or_timestamp_churn():
    root = stale_repo()
    assert W.run(root)["status"] == "local_complete"
    first = watched(root)
    records = {rid: r["timestamp"] for rid, r in S.load_dataset(root).items()}
    for _ in range(2):
        assert W.run(root)["status"] == "local_complete"
        assert watched(root) == first                                # byte-identical incl. generated_at / last_updated
    assert {rid: r["timestamp"] for rid, r in S.load_dataset(root).items()} == records


def test_unchanged_records_keep_their_timestamps_when_one_event_is_added():
    root = make_repo()
    assert W.run(root)["status"] == "local_complete"
    publish(root, "export")
    before = S.load_dataset(root)
    with (root / ".agent/learning/events.jsonl").open("a") as f:
        f.write(json.dumps({"event_type": "PROCEDURE_OUTCOME", "procedure": "p2", "outcome": "success",
                            "timestamp": "2026-01-02T00:00:00Z"}) + "\n")
    publish(root, "event")
    assert W.run(root)["status"] == "local_complete"
    after = S.load_dataset(root)
    changed = {rid for rid in after if rid in before and after[rid]["sha256"] != before[rid]["sha256"]}
    assert changed <= {"events__PROCEDURE_OUTCOME__001", "soul_index"}
    for rid in set(before) - changed - {"soul_index"}:
        assert after[rid]["timestamp"] == before[rid]["timestamp"], rid


def test_bootstrap_without_state_and_current_export_returns_nothing_and_writes_nothing():
    root = make_repo()
    assert W.run(root)["status"] == "local_complete"
    publish(root, "export")
    before = watched(root)
    result = W.bootstrap(root)
    assert result and result["status"] == "local_complete"
    assert watched(root) == before
