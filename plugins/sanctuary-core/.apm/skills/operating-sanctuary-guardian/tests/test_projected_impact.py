"""MISSION-031 tests: projected impact, overlap matrix, manifest-driven leases (flight board).

Purpose: a mission checked out WITHOUT --targets must lock exactly what its manifest declares; overlapping missions are refused, disjoint ones
    are dispatchable together; globs are handled conservatively (never under-lock); regenerated shared files are merge notes, not lease conflicts;
    declared impact can be verified against what a worktree actually changed; overlap checks are fast.
Key Input Dependencies: pytest; scripts/manage_coordination.py, scripts/impact.py, scripts/relay.py; git for the verify tests.
Key Functions: test_checkout_without_targets_locks_from_manifest, test_overlapping_manifest_checkout_is_blocked, test_checkout_without_declaration_is_refused,
    test_paths_overlap_rules, test_shared_generated_files_are_not_locks, test_manual_targets_extend_the_manifest, test_matrix_and_suggest,
    test_suggest_respects_active_leases, test_verify_flags_undeclared_edits, test_checkout_marks_manifest_in_worktree_not_main_tree,
    test_cli_exit_codes, test_validate_impact, test_relay_surfaces_dispatchable, test_overlap_latency.
Usage: python3 -m pytest plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_projected_impact.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import manage_coordination as mc  # noqa: E402
import impact  # noqa: E402
import relay  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import impact_eval as ev  # noqa: E402


def write_missions(root: Path, missions: dict) -> None:
    d = root / "LEARNING" / "missions"
    d.mkdir(parents=True, exist_ok=True)
    for mid, spec in missions.items():
        body = {"id": mid, "title": f"Mission {mid}", "status": "PROPOSED", "priority_score": spec.get("score", 100.0)}
        if "impact" in spec:
            body["projected_impact"] = spec["impact"]
        (d / f"{mid}.json").write_text(json.dumps(body))


@pytest.fixture
def root(tmp_path):
    write_missions(tmp_path, {
        "MISSION-A": {"impact": {"paths": ["plugins/x/scripts/a.py", "docs/a/"], "resources": ["port:8080"]}},
        "MISSION-B": {"impact": {"paths": ["plugins/x/scripts/a.py"]}},                    # same file as A
        "MISSION-C": {"impact": {"paths": ["plugins/y/"]}},                                 # disjoint from A and B
        "MISSION-D": {"impact": {"paths": ["docs/a/guide.md"]}},                            # inside A's directory
        "MISSION-E": {},                                                                    # no declaration
    })
    return tmp_path


def test_checkout_without_targets_locks_from_manifest(root):
    res = mc.checkout(root, "MISSION-A", "agent-1")
    assert res["status"] == "success", res
    assert sorted(res["locked_targets"]) == ["docs/a/", "plugins/x/scripts/a.py"]
    assert res["exclusive_resources"] == ["port:8080"]


def test_overlapping_manifest_checkout_is_blocked(root):
    assert mc.checkout(root, "MISSION-A", "agent-1")["status"] == "success"
    blocked = mc.checkout(root, "MISSION-B", "agent-2")          # no --targets: the manifest must supply them
    assert blocked["status"] == "conflict", f"silent conflict: {blocked}"
    inside_dir = mc.checkout(root, "MISSION-D", "agent-2")
    assert inside_dir["status"] == "conflict", f"silent conflict: {inside_dir}"
    assert mc.checkout(root, "MISSION-C", "agent-2")["status"] == "success"   # disjoint


def test_checkout_without_declaration_is_refused(root):
    res = mc.checkout(root, "MISSION-E", "agent-1")               # no manifest impact and no --targets
    assert res["status"] == "error" and "projected_impact" in res["error"], res


# ---- GREEN phase ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("a,b,expected", [
    ("a/b.py", "a/b.py", True), ("a/", "a/b.py", True), ("a/b.py", "a/", True), ("a/b.py", "a/c.py", False), ("a", "ab", False),
    ("a/b.py/x", "a/b.py", False),                                    # a file has no children
    ("plugins/*/scripts/", "plugins/x/scripts/a.py", True), ("plugins/*/scripts/", "plugins/x/docs/a.py", False),
    ("**/*.md", "docs/a.md", True), ("**/*.md", "docs/a.py", False), ("**/*.md", "docs/", True),
    ("plugins/**", "plugins/x/y/z.py", True), ("plugins/**/a.py", "plugins/x/y/a.py", True), ("plugins/**/a.py", "plugins/x/y/b.py", False),
    ("docs/*.md", "docs/sub/a.md", False), ("docs/**/*.md", "docs/sub/a.md", True),
    ("LICENSE", "LICENSE/x", True),                                   # extension-less names may be directories: ambiguity is overlap
    ("a/*/c", "a/b/", True), ("*.py", "x.md", False), ("*.py", "y.py", True), ("./a//b.py", "a/b.py", True), ("", "a", False),
])
def test_paths_overlap_rules(a, b, expected):
    assert impact.paths_overlap(a, b) is expected and impact.paths_overlap(b, a) is expected  # symmetric


def test_shared_generated_files_are_not_locks(tmp_path):
    write_missions(tmp_path, {"M-1": {"impact": {"paths": ["x/a.py", "LEARNING/index.jsonl", ".agent/learning/cognitive_ledger.json"]}},
                              "M-2": {"impact": {"paths": ["y/b.py", "LEARNING/index.jsonl"]}}})
    first = mc.checkout(tmp_path, "M-1", "a")
    assert first["status"] == "success" and first["locked_targets"] == ["x/a.py"]
    assert "LEARNING/index.jsonl" in first["regenerate_after_merge"]
    assert mc.checkout(tmp_path, "M-2", "b")["status"] == "success"      # both touch index.jsonl: regenerate, not a conflict
    assert not impact.conflicts(impact.load_manifest_impact(tmp_path, "M-1"), impact.load_manifest_impact(tmp_path, "M-2"))


def test_manual_targets_extend_the_manifest(root):
    res = mc.checkout(root, "MISSION-C", "a", targets=["extra/file.py"])
    assert res["impact_source"] == "manifest+manual" and sorted(res["locked_targets"]) == ["extra/file.py", "plugins/y/"]
    assert mc.checkout(root, "MISSION-E", "b", targets=["only/manual.py"])["impact_source"] == "manual"
    assert mc.checkout(root, "MISSION-E", "c", allow_unlocked=True)["impact_source"] == "unlocked"


def test_matrix_and_suggest(root):
    write_missions(root, {"MISSION-A": {"score": 100, "impact": {"paths": ["plugins/x/scripts/a.py", "docs/a/"], "resources": ["port:8080"]}},
                          "MISSION-B": {"score": 90, "impact": {"paths": ["plugins/x/scripts/a.py"]}},
                          "MISSION-C": {"score": 80, "impact": {"paths": ["plugins/y/"]}},
                          "MISSION-D": {"score": 70, "impact": {"paths": ["docs/a/guide.md"]}},
                          "MISSION-F": {"score": 60, "impact": {"paths": ["other/"], "resources": ["port:8080"]}},   # shares a resource with A
                          "MISSION-E": {"score": 50}})
    m = mc.build_matrix(root)
    assert m["conflicts"]["MISSION-A"] == ["MISSION-B", "MISSION-D", "MISSION-F"] and m["conflicts"]["MISSION-C"] == [] and m["unregistered"] == ["MISSION-E"]
    s = mc.suggest(root, count=3)
    assert s["selected"] == ["MISSION-A", "MISSION-C"]                      # greedy by priority: B, D, F are blocked by A
    assert {b["id"]: b["by"] for b in s["blocked"]} == {"MISSION-B": "MISSION-A", "MISSION-D": "MISSION-A", "MISSION-F": "MISSION-A"}
    assert s["unregistered"] == ["MISSION-E"]


def test_suggest_respects_active_leases(tmp_path):
    root = tmp_path
    write_missions(root, {"MISSION-A": {"score": 100, "impact": {"paths": ["docs/a/"]}}, "MISSION-C": {"score": 80, "impact": {"paths": ["plugins/y/"]}}})
    mc.checkout(root, "MISSION-X-HELD", "someone", targets=["plugins/y/sub/file.py"])
    s = mc.suggest(root, count=3)
    assert s["selected"] == ["MISSION-A"] and s["blocked"] == [{"id": "MISSION-C", "by": "MISSION-X-HELD"}]


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, capture_output=True, text=True, check=True)


def test_verify_flags_undeclared_edits(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    for f in ("plugins/x/a.py", "plugins/y/b.py", "docs/c.md", "LEARNING/index.jsonl"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("base\n")
    _git(tmp_path, "add", "."), _git(tmp_path, "commit", "-q", "-m", "base"), _git(tmp_path, "checkout", "-q", "-b", "feat")
    write_missions(tmp_path, {"MISSION-A": {"impact": {"paths": ["plugins/x/"]}}})
    (tmp_path / "plugins/x/a.py").write_text("changed\n")                               # inside the declaration
    (tmp_path / "LEARNING/index.jsonl").write_text("regenerated\n")             # shared generated: allowed
    _git(tmp_path, "commit", "-qam", "work")
    ok = mc.verify_impact(tmp_path, "MISSION-A", worktree=".", base="main")
    assert ok["ok"] and ok["undeclared"] == []
    (tmp_path / "plugins/y/b.py").write_text("sneaky\n")                                # committed outside? no: uncommitted, outside
    (tmp_path / "docs/new.md").write_text("untracked\n")                                # untracked, outside
    bad = mc.verify_impact(tmp_path, "MISSION-A", worktree=".", base="main")
    assert not bad["ok"] and bad["undeclared"] == ["docs/new.md", "plugins/y/b.py"]


def test_checkout_marks_manifest_in_worktree_not_main_tree(tmp_path):
    write_missions(tmp_path, {"MISSION-A": {"impact": {"paths": ["x/"]}}})
    wt = tmp_path / ".worktrees" / "wt-a"
    write_missions(wt, {"MISSION-A": {"impact": {"paths": ["x/"]}}})                       # the worktree's own copy of the manifest
    main_copy = (tmp_path / "LEARNING/missions/MISSION-A.json").read_text()
    assert mc.checkout(tmp_path, "MISSION-A", "a", worktree=".worktrees/wt-a")["status"] == "success"
    assert (tmp_path / "LEARNING/missions/MISSION-A.json").read_text() == main_copy, "main tree manifest was modified"
    assert json.loads((wt / "LEARNING/missions/MISSION-A.json").read_text())["status"] == "IN_PROGRESS"


def test_cli_exit_codes(root):
    script = str(SCRIPTS / "manage_coordination.py")
    env = {**os.environ, "PROJECT_ROOT": str(root)}
    run = lambda *a: subprocess.run([sys.executable, script, *a], env=env, capture_output=True, text=True)  # noqa: E731
    assert run("checkout", "--mission", "MISSION-A", "--agent", "a").returncode == 0
    refused = run("checkout", "--mission", "MISSION-E", "--agent", "b")
    assert refused.returncode == 1 and "REFUSED" in refused.stdout
    blocked = run("checkout", "--mission", "MISSION-B", "--agent", "b", "--format", "json")
    assert blocked.returncode == 1 and json.loads(blocked.stdout)["status"] == "conflict"   # JSON mode must fail too
    assert "MISSION-C" in run("suggest").stdout and run("matrix").returncode == 0


def test_validate_impact():
    assert impact.validate_impact(None) == [] and impact.validate_impact({"paths": ["a/b.py"], "resources": ["p:1"]}) == []
    assert impact.validate_impact({"paths": []}) and impact.validate_impact({"resources": []}) and impact.validate_impact("x")
    assert impact.validate_impact({"paths": ["/abs/path"]}) and impact.validate_impact({"paths": ["../escape"]})
    assert impact.validate_impact({"paths": ["a"], "resources": "no"})


def test_relay_surfaces_dispatchable(tmp_path):
    root = tmp_path
    write_missions(root, {"MISSION-A": {"score": 100, "impact": {"paths": ["docs/a/"]}}, "MISSION-C": {"score": 80, "impact": {"paths": ["plugins/y/"]}}})
    text = relay.render_markdown(relay.build_relay(root))
    assert "Dispatchable in parallel now" in text and "MISSION-A" in text and "MISSION-C" in text


def test_overlap_latency(tmp_path):
    """Prediction 3 (synthetic scale): 200 missions x 20 active leases, matrix and per-mission lease check each under 500 ms."""
    import random
    rng = random.Random(3)
    pool = [f"plugins/p{i}/{d}/" for i in range(12) for d in ("scripts", "docs", "tests")] + [f"area{i}/**/*.md" for i in range(8)]
    impacts = {f"M-{i}": {"paths": rng.sample(pool, 3), "resources": []} for i in range(200)}
    t0 = time.perf_counter()
    impact.overlap_matrix(impacts)
    matrix_s = time.perf_counter() - t0
    leases = [{"mission_id": f"L-{i}", "locked_targets": rng.sample(pool, 3), "exclusive_resources": []} for i in range(20)]
    t0 = time.perf_counter()
    for imp in impacts.values():
        any(impact.conflicts(imp, impact.lease_impact(l)) for l in leases)
    per_mission_ms = (time.perf_counter() - t0) / 200 * 1000
    assert matrix_s < 0.5, f"matrix for 200 missions took {matrix_s:.3f}s"
    assert per_mission_ms < 500, f"lease check took {per_mission_ms:.1f}ms per mission"


# ---- evaluation harness and recorded evidence -------------------------------------------------------------------------
def test_eval_ground_truth_is_independent_and_correct():
    u = ev.build_universe()
    assert ev.expand("plugins/p1/docs/", u) == {f for f in u if f.startswith("plugins/p1/docs/")}
    assert ev.expand("area0/f1.md", u) == {"area0/f1.md"}
    assert ev.expand("**/*.md", u) == {f for f in u if f.endswith(".md")}
    assert ev.expand("plugins/*/docs/", u) == {f for f in u if f.startswith("plugins/p") and "/docs/" in f}
    assert ev.expand("area1/**/g?.md", u) == {"area1/sub/g1.md"} and ev.expand("nothing/here", u) == set()


def test_eval_quick_run_has_no_false_negatives():
    stats = ev.run_e1(60, baseline_rev=None, seed=7)
    assert stats["new_false_negatives"] == 0 and stats["manual_targets_needed"] == 0 and stats["overlapping_truth"] > 0


RESULTS = Path(__file__).resolve().parents[6] / "LEARNING" / "experiments" / "mission-031" / "results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="evaluation results not recorded (run impact_eval.py --out ...)")
def test_recorded_results_meet_the_pre_registered_rule():
    """The stored evidence must satisfy the decision rule written before the runs (and the baseline must show the defect)."""
    r = json.loads(RESULTS.read_text())
    e1, e2, e3, e4 = r["e1_random_pairs"], r["e2_real_git"], r["e3_latency"], r["e4_live_registry"]
    assert e1["old_silent_conflict_rate"] == 1.0 and e1["overlapping_truth"] > 0                      # prediction 1: the defect existed
    assert e1["new_false_negatives"] == 0 and e1["manual_targets_needed"] == 0 and e1["new_false_positive_rate"] <= 0.30   # prediction 2
    assert e3["registry_matrix_plus_suggest_ms_p95"] <= 100 and e3["synthetic_200_missions_20_leases_per_mission_check_ms_p95"] <= 500   # prediction 3
    assert e2["predicted_disjoint_real_conflicts"] == 0 and e2["real_conflicts_not_predicted"] == 0                                   # prediction 4
    assert e2["undeclared_planted"] > 0 and e2["undeclared_flagged"] == e2["undeclared_planted"] and e2["declared_edits_unflagged"] == 0
    assert len(e4["suggested"]) == 3 and e4["mutually_disjoint"]                                                                      # prediction 5
