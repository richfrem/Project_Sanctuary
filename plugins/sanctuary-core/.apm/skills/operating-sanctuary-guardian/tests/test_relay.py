"""MISSION-025 tests: predecessor relay, checkin --handoff, wakeup integration, and the evaluation graders.

Purpose: verify the relay is built from existing ledgers, stays small, round-trips a handoff note from checkin to the briefing, never breaks
    the wakeup, and that the A/B evaluation's graders and answer key are strict and independent of the relay code.
Key Input Dependencies: pytest; scripts/relay.py, scripts/manage_coordination.py, scripts/guardian_wakeup.py, tests/relay_eval.py.
Key Functions: test_vendor_of, test_build_relay_orders_and_clips, test_older_handovers_are_shorter, test_next_step_at_the_end_survives_for_recent_handoffs, test_established_pointers, test_open_work_and_inbox, test_empty_project,
    test_render_budget, test_checkin_handoff_normalisation, test_handoff_round_trip, test_wakeup_includes_relay, test_wakeup_survives_relay_failure,
    test_graders_are_strict, test_synthetic_key_matches_snapshot, test_recorded_ab_results_match_the_reported_numbers (skips until recorded).
Usage: python3 -m pytest plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_relay.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))
import relay  # noqa: E402
import manage_coordination as mc  # noqa: E402
import guardian_wakeup as gw  # noqa: E402
import relay_eval as ev  # noqa: E402


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def project(tmp_path):
    write_jsonl(tmp_path / ".agent/coordination/leases.jsonl", [
        {"event_type": "CHECKOUT", "agent_id": "codex", "mission_id": "M-1"},
        {"event_type": "CHECKIN", "agent_id": "codex", "mission_id": "MISSION-901", "outcome_status": "COMPLETED", "pr_number": 7,
         "timestamp": "2026-11-01T09:00:00+00:00", "handoff": "Rejected approach X. " + "x" * 400},
        {"event_type": "CHECKIN", "agent_id": "antigravity", "mission_id": "MISSION-902", "outcome_status": "ABANDONED", "pr_number": None,
         "timestamp": "2026-11-02T10:00:00+00:00"},
        {"event_type": "CHECKIN", "agent_id": "someone-new", "mission_id": "MISSION-903", "outcome_status": "COMPLETED", "pr_number": 9,
         "timestamp": "2026-11-03T11:00:00+00:00", "handoff": "  line one\n\n  line   two "},
    ])
    mdir = tmp_path / "LEARNING/research/missions"
    mdir.mkdir(parents=True)
    (mdir / "MISSION-901.json").write_text(json.dumps({"id": "MISSION-901", "status": "RESOLVED", "verdict": "CONFIRMED", "updated_at": "2026-11-01",
        "evidence": [{"citation": "LEARNING/research/experiments/mission-901/README.md"}]}))
    (mdir / "MISSION-902.json").write_text(json.dumps({"id": "MISSION-902", "status": "RESOLVED", "verdict": "FALSIFIED", "updated_at": "2026-11-02",
        "evidence": ["free text with no path"]}))
    (mdir / "MISSION-910.json").write_text(json.dumps({"id": "MISSION-910", "status": "RESOLVED", "verdict": "CORROBORATED", "updated_at": "2026-11-03",
        "evidence": [{"citation": ""}]}))
    (tmp_path / "LEARNING/research/experiments/mission-910").mkdir(parents=True)
    (tmp_path / "LEARNING/research/experiments/mission-910/README.md").write_text("log")
    write_jsonl(tmp_path / "LEARNING/research/index.jsonl", [
        {"id": "MISSION-901", "status": "RESOLVED", "priority_score": 900}, {"id": "MISSION-920", "status": "PROPOSED", "priority_score": 50},
        {"id": "MISSION-921", "status": "PROPOSED", "priority_score": 300}, {"id": "MISSION-922", "status": "IN_PROGRESS", "priority_score": 200},
        {"id": "MISSION-923", "status": "PROPOSED", "priority_score": 10}])
    write_jsonl(tmp_path / "LEARNING/research/inbox.jsonl", [{"idea": "a"}, {"idea": "b"}])
    return tmp_path


def test_vendor_of():
    assert relay.vendor_of("claude-code") == "Anthropic" and relay.vendor_of("antigravity") == "Google"
    assert relay.vendor_of("Codex") == "OpenAI" and relay.vendor_of("mlx") == "local Qwen" and relay.vendor_of("zzz") == "unknown"


def test_build_relay_orders_and_clips(project):
    r = relay.build_relay(project)
    assert [h["mission"] for h in r["handovers"]] == ["MISSION-903", "MISSION-902", "MISSION-901"]  # newest first; CHECKOUT ignored
    assert r["handovers"][0]["handoff"] == "line one line two"
    assert len(r["handovers"][2]["handoff"]) <= relay.MAX_HANDOFF_SHOWN and r["handovers"][2]["handoff"].endswith("…")  # newest 3 get the long cap
    r5 = relay.build_relay(project, handovers=5)  # (only 3 events here) the older-cap rule is covered in test_older_handovers_are_shorter
    assert r["handovers"][1]["handoff"] is None and r["handovers"][1]["pr"] is None
    assert r["handovers"][2]["vendor"] == "OpenAI" and r["handovers"][0]["vendor"] == "unknown"


def test_older_handovers_are_shorter(tmp_path):
    write_jsonl(tmp_path / ".agent/coordination/leases.jsonl", [
        {"event_type": "CHECKIN", "agent_id": "codex", "mission_id": f"M-{i}", "outcome_status": "COMPLETED", "pr_number": i,
         "timestamp": f"2026-11-0{i}T09:00:00+00:00", "handoff": "z" * 500} for i in range(1, 6)])
    caps = [len(h["handoff"]) for h in relay.build_relay(tmp_path)["handovers"]]  # newest first
    assert caps == [relay.MAX_HANDOFF_SHOWN] * relay.FULL_HANDOFFS + [relay.MAX_HANDOFF_OLDER] * 2


def test_next_step_at_the_end_survives_for_recent_handoffs(tmp_path):
    note = "Rejected the SQLite approach because it locks. Benchmark BENCH-1 already run, cite it. Next: run the ZETA-7 regression before merging."
    write_jsonl(tmp_path / ".agent/coordination/leases.jsonl", [{"event_type": "CHECKIN", "agent_id": "codex", "mission_id": "M", "outcome_status": "COMPLETED",
                                                                   "pr_number": 1, "timestamp": "2026-11-01T09:00:00+00:00", "handoff": note}])
    assert "ZETA-7" in relay.render_markdown(relay.build_relay(tmp_path))


def test_established_pointers(project):
    est = {e["id"]: e for e in relay.build_relay(project)["established"]}
    assert est["MISSION-901"]["pointer"] == "LEARNING/research/experiments/mission-901/README.md"        # path citation
    assert est["MISSION-910"]["pointer"] == "LEARNING/research/experiments/mission-910/README.md"        # experiment log fallback
    assert est["MISSION-902"]["pointer"] == "LEARNING/research/missions/MISSION-902.json"                # manifest fallback
    assert [e["id"] for e in relay.build_relay(project)["established"]][0] == "MISSION-910"              # newest update first


def test_open_work_and_inbox(project):
    r = relay.build_relay(project)
    assert [o["id"] for o in r["open"]] == ["MISSION-921", "MISSION-922", "MISSION-920"] and r["inbox_items"] == 2


def test_empty_project(tmp_path):
    r = relay.build_relay(tmp_path)
    assert r == {"handovers": [], "established": [], "open": [], "inbox_items": 0, "dispatchable": []}
    assert "When you finish" in relay.render_markdown(r)


def test_render_budget(project):
    r = relay.build_relay(project, handovers=5, results=10)
    r["handovers"] = [{"when": "2026-11-03 16:00", "agent": "claude-code", "vendor": "Anthropic", "mission": "MISSION-027", "outcome": "COMPLETED",
                       "pr": 289, "handoff": "y" * (relay.MAX_HANDOFF_SHOWN if i < relay.FULL_HANDOFFS else relay.MAX_HANDOFF_OLDER)} for i in range(5)]
    r["established"] = [{"id": f"MISSION-0{i:02d}", "verdict": "CORROBORATED", "pointer": f"LEARNING/research/experiments/mission-0{i:02d}/README.md"}
                        for i in range(10)]
    r["dispatchable"] = ["MISSION-031", "MISSION-023", "MISSION-030"]
    text = relay.render_markdown(r)
    assert len(text.splitlines()) <= 40 and len(text) <= 3500, (len(text.splitlines()), len(text))
    assert "no handoff note" in relay.render_markdown(relay.build_relay(project))


def test_checkin_handoff_normalisation():
    assert mc.normalize_handoff(None) is None and mc.normalize_handoff("   \n ") is None
    assert mc.normalize_handoff(" a \n b ") == "a b"
    long = mc.normalize_handoff("w " * 1000)
    assert len(long) <= mc.MAX_HANDOFF_CHARS and long.endswith("…")


def test_handoff_round_trip(tmp_path):
    mc.checkout(tmp_path, "MISSION-X", "codex", targets=["a.py"])
    res = mc.checkin(tmp_path, "MISSION-X", "codex", status="COMPLETED", pr=12, handoff="Do not retry plan B.  Next: run T-9.")
    assert res["handoff"] == "Do not retry plan B. Next: run T-9."
    events = [json.loads(l) for l in (tmp_path / ".agent/coordination/leases.jsonl").read_text().splitlines()]
    assert events[-1]["handoff"] == "Do not retry plan B. Next: run T-9."
    h = relay.build_relay(tmp_path)["handovers"][0]
    assert h["agent"] == "codex" and h["pr"] == 12 and "run T-9" in h["handoff"]
    mc.checkout(tmp_path, "MISSION-Y", "codex", targets=["b.py"])
    mc.checkin(tmp_path, "MISSION-Y", "codex")  # no handoff: the key is absent, as before this feature
    assert "handoff" not in json.loads((tmp_path / ".agent/coordination/leases.jsonl").read_text().splitlines()[-1])


def _wakeup_root(tmp_path: Path, project: Path) -> Path:
    for rel in ("context", "START_HERE.md", "AGENTS.md"):
        src = Path(__file__).resolve().parents[6] / rel
        if src.is_dir():
            subprocess.run(["cp", "-R", str(src), str(project / rel)], check=True)
        elif src.exists():
            (project / rel).write_text(src.read_text())
    return project


def test_wakeup_includes_relay(tmp_path, project):
    root = _wakeup_root(tmp_path, project)
    out = subprocess.run([sys.executable, str(SCRIPTS / "guardian_wakeup.py"), "--format", "markdown"], cwd=root,
                         env={**os.environ, "PROJECT_ROOT": str(root)}, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-300:]
    assert "Predecessor Relay" in out.stdout and "MISSION-903" in out.stdout and "line one line two" in out.stdout
    assert out.stdout.index("III. Pre-Flight Brief") < out.stdout.index("Predecessor Relay") < out.stdout.index("IV. Layer 1")


def test_wakeup_survives_relay_failure(project, monkeypatch):
    monkeypatch.setattr(relay, "build_relay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert gw.get_relay_section(project) == ""


def test_graders_are_strict():
    g = ev.grade
    assert g(ev.q("a", "agent", "", "", "codex"), "codex") and not g(ev.q("a", "agent", "", "", "codex"), "codex or antigravity")
    assert g(ev.q("a", "id", "", "", "HF-SYNC-4"), "HF-SYNC-4") and not g(ev.q("a", "id", "", "", "HF-SYNC-4"), "HF-SYNC-4 and MISSION-027")
    assert g(ev.q("a", "number", "", "", 290), "PR 290") and not g(ev.q("a", "number", "", "", 290), "289 or 290")
    assert g(ev.q("a", "verdict", "", "", "CONFIRMED"), "confirmed") and not g(ev.q("a", "verdict", "", "", "CONFIRMED"), "CONFIRMED / FALSIFIED")
    vp = ev.q("a", "verdict_path", "", "", {"verdict": "FALSIFIED", "path": "LEARNING/x/README.md"})
    assert g(vp, "FALSIFIED | LEARNING/x/README.md") and not g(vp, "FALSIFIED") and not g(vp, "CORROBORATED | LEARNING/x/README.md")
    assert g(ev.q("a", "yesno", "", "", "no"), "No.") and not g(ev.q("a", "yesno", "", "", "no"), "yes")
    assert not g(ev.q("a", "token", "", "", "ZETA-7"), None) and g(ev.q("a", "token", "", "", "ZETA-7"), "run zeta-7")


def test_synthetic_key_matches_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "main_root", lambda: Path(__file__).resolve().parents[6])
    ev.build_snapshot_synthetic(tmp_path)
    events = [json.loads(l) for l in (tmp_path / ".agent/coordination/leases.jsonl").read_text().splitlines()]
    blob = json.dumps(events)
    for item in ev.questions_synthetic():
        if item["kind"] in ("token", "number"):
            assert str(item["expected"]).lower() in blob.lower(), f"{item['id']}: key not present in the snapshot ledger"
    assert relay.build_relay(tmp_path)["handovers"][0]["mission"] == "MISSION-903"


# ---- recorded A/B evidence vs the pre-registered decision rule (MISSION-025) ------------------------------------------
EXPERIMENT = Path(__file__).resolve().parents[6] / "LEARNING" / "research" / "experiments" / "mission-025"


@pytest.mark.skipif(not (EXPERIMENT / "results_control.json").exists() or not (EXPERIMENT / "results_treatment.json").exists(),
                    reason="A/B results not recorded (run relay_eval.py: prepare, digest, run for both labels)")
def test_recorded_ab_results_match_the_reported_numbers():
    """The summaries stored with the experiment must agree with their own rows (guards against hand-edited or stale numbers)."""
    for label in ("control", "treatment"):
        data = json.loads((EXPERIMENT / f"results_{label}.json").read_text())
        recomputed = ev.summarize(data["rows"])
        assert recomputed == data["summary"], f"{label}: stored summary does not match its rows"
        assert len(data["rows"]) == 22 and {r["id"][0] for r in data["rows"]} == {"A", "B"}
    control = json.loads((EXPERIMENT / "results_control.json").read_text())["summary"]
    treatment = json.loads((EXPERIMENT / "results_treatment.json").read_text())["summary"]
    assert treatment["correct_rate"] > control["correct_rate"], "the relay did not improve correctness"
    assert treatment["mean_tool_calls"] < control["mean_tool_calls"], "the relay did not reduce discovery work"
