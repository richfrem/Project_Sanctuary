"""Tests for merge_prep.py (MISSION-034): regenerate shared generated files, report (never hide) ledger drift."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import merge_prep  # noqa: E402

REPO = HERE.parents[6]


def _assessment(n, agent="claude-code", raw=12):
    return {"event_type": "CONTRIBUTION_ASSESSED", "assessment_id": f"id-{n}", "timestamp": f"2026-11-04T10:{n:02d}:00+00:00",
            "task_id": f"T-{n}", "agent_id": agent, "raw_score": raw, "cognitive_fitness": round(raw / 15, 3),
            "dimensions": {"useful_outcome": 3, "learning_depth": 3, "verification_rigor": 2,
                           "token_economy": 2, "anti_asch_integrity": 2}, "notes": f"n{n}"}


def _make_root(n_events=3, with_missions=True):
    root = Path(tempfile.mkdtemp())
    learn = root / ".agent" / "learning"
    learn.mkdir(parents=True)
    events = [_assessment(i) for i in range(n_events)]
    (learn / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    if with_missions:
        md = root / "LEARNING" / "missions"
        md.mkdir(parents=True)
        for name in ("MISSION-033.json", "MISSION-034.json"):
            shutil.copy(REPO / "LEARNING" / "missions" / name, md / name)
    return root, events


class TestLedger(unittest.TestCase):
    def test_implied_matches_live_scorer_arithmetic(self):
        root, events = _make_root(3)
        implied = merge_prep.implied_ledger(root)
        self.assertEqual(implied["total_assessments"], 3)
        self.assertEqual(implied["cumulative_score"], 36)
        self.assertEqual(implied["recent_assessments"][0]["assessment_id"], "id-2")  # newest first

    def test_rebuild_is_independent_of_line_order(self):
        root, events = _make_root(4)
        a = merge_prep.implied_ledger(root)
        (root / ".agent" / "learning" / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in reversed(events)))
        b = merge_prep.implied_ledger(root)
        self.assertEqual(a, b)

    def test_status_ok_stale_drift(self):
        root, _ = _make_root(3)
        implied = merge_prep.implied_ledger(root)
        self.assertEqual(merge_prep.ledger_status(json.loads(json.dumps(implied)), implied), "ok")
        stale = json.loads(json.dumps(implied)); stale["total_assessments"] = 2; stale["cumulative_score"] = 24
        self.assertEqual(merge_prep.ledger_status(stale, implied), "stale")
        drift = json.loads(json.dumps(implied)); drift["total_assessments"] = 4; drift["cumulative_score"] = 50
        self.assertEqual(merge_prep.ledger_status(drift, implied), "drift")

    def test_updated_at_is_ignored_for_equality(self):
        root, _ = _make_root(2)
        implied = merge_prep.implied_ledger(root)
        other = json.loads(json.dumps(implied)); other["updated_at"] = "2099-01-01T00:00:00+00:00"
        self.assertEqual(merge_prep.ledger_status(other, implied), "ok")

    def test_hunk_inside_json_keeps_shared_lines(self):
        root, _ = _make_root(2)
        text = ('{\n  "schema_version": 1,\n<<<<<<< HEAD\n  "total_assessments": 3,\n=======\n'
                '  "total_assessments": 2,\n>>>>>>> b\n  "cumulative_score": 24\n}\n')
        p = root / "ledger.json"; p.write_text(text)
        sides = merge_prep.read_stored_sides(p)
        self.assertEqual([(s["schema_version"], s["total_assessments"], s["cumulative_score"]) for s in sides],
                         [(1, 3, 24), (1, 2, 24)])

    def test_conflicted_ledger_sides_are_both_read(self):
        root, _ = _make_root(2)
        ours = {"total_assessments": 3, "cumulative_score": 36, "agents": {}}
        theirs = {"total_assessments": 2, "cumulative_score": 24, "agents": {}}
        text = ("<<<<<<< HEAD\n" + json.dumps(ours, indent=1) + "\n=======\n"
                + json.dumps(theirs, indent=1) + "\n>>>>>>> branch\n")
        p = root / "ledger.json"; p.write_text(text)
        sides = merge_prep.read_stored_sides(p)
        self.assertEqual([s["total_assessments"] for s in sides], [3, 2])


class TestRegenerate(unittest.TestCase):
    def test_stale_ledger_is_rebuilt_from_events(self):
        root, _ = _make_root(3)
        led = root / ".agent" / "learning" / "cognitive_ledger.json"
        led.write_text(json.dumps({"total_assessments": 1, "cumulative_score": 12, "agents": {}}))
        rep = merge_prep.regenerate(root)
        self.assertEqual(rep["ledger"]["status"], "stale")
        self.assertEqual(rep["ledger"]["action"], "rebuilt")
        self.assertEqual(json.loads(led.read_text())["total_assessments"], 3)
        self.assertFalse(rep["drift"])

    def test_drift_is_reported_and_file_untouched(self):
        root, _ = _make_root(3)
        led = root / ".agent" / "learning" / "cognitive_ledger.json"
        orphan = merge_prep.implied_ledger(root)
        orphan["total_assessments"] = 4; orphan["cumulative_score"] = 51
        led.write_text(json.dumps(orphan))
        before = led.read_text()
        rep = merge_prep.regenerate(root)
        self.assertTrue(rep["drift"])
        self.assertEqual(rep["ledger"]["action"], "left-unchanged")
        self.assertEqual(led.read_text(), before)

    def test_conflicted_ledger_with_one_drifting_side_is_not_overwritten(self):
        root, _ = _make_root(2)
        led = root / ".agent" / "learning" / "cognitive_ledger.json"
        orphan = {"total_assessments": 3, "cumulative_score": 40, "agents": {}}
        ok = {"total_assessments": 2, "cumulative_score": 24, "agents": {}}
        text = "<<<<<<< HEAD\n" + json.dumps(orphan) + "\n=======\n" + json.dumps(ok) + "\n>>>>>>> b\n"
        led.write_text(text)
        rep = merge_prep.regenerate(root)
        self.assertTrue(rep["drift"])
        self.assertEqual(led.read_text(), text)

    def test_conflicted_ledger_without_drift_is_rebuilt(self):
        root, _ = _make_root(3)
        led = root / ".agent" / "learning" / "cognitive_ledger.json"
        a = {"total_assessments": 2, "cumulative_score": 24, "agents": {}}
        b = {"total_assessments": 3, "cumulative_score": 36, "agents": {}}
        led.write_text("<<<<<<< HEAD\n" + json.dumps(a) + "\n=======\n" + json.dumps(b) + "\n>>>>>>> b\n")
        rep = merge_prep.regenerate(root)
        self.assertFalse(rep["drift"])
        self.assertEqual(json.loads(led.read_text())["total_assessments"], 3)

    def test_backlog_is_regenerated_and_idempotent(self):
        root, _ = _make_root(1)
        merge_prep.regenerate(root)
        idx = root / "LEARNING" / "research" / "index.jsonl"
        first = idx.read_bytes()
        self.assertEqual(len(first.splitlines()), 2)
        merge_prep.regenerate(root)
        self.assertEqual(idx.read_bytes(), first)

    def test_chronicle_ledger_is_regenerated_from_entries(self):
        root, _ = _make_root(1, with_missions=False)
        ent = root / "chronicle" / "entries"
        ent.mkdir(parents=True)
        for n in (1, 2):
            (ent / f"{n:03d}_x.md").write_text(f"**Title:** T{n}\n**Date:** 2026-11-04\n**Status:** draft\n\n---\nb\n")
        (root / "chronicle" / "ledger.json").write_text('{"total_entries": 99, "entries": []}')
        rep = merge_prep.regenerate(root)
        self.assertIn("chronicle/ledger.json", rep["generated"])
        led = json.loads((root / "chronicle" / "ledger.json").read_text())
        self.assertEqual((led["total_entries"], led["entries"][0]["id"]), (2, 2))

    def test_regenerate_is_deterministic_across_runs(self):
        a, _ = _make_root(3)
        b, _ = _make_root(3)
        merge_prep.regenerate(a); merge_prep.regenerate(b)
        for rel in (".agent/learning/cognitive_ledger.json", "LEARNING/research/index.jsonl", "LEARNING/backlog.md"):
            self.assertEqual((a / rel).read_bytes(), (b / rel).read_bytes(), rel)


class TestAttributes(unittest.TestCase):
    def test_missing_union_rules_listed(self):
        root, _ = _make_root(1, with_missions=False)
        self.assertEqual(set(merge_prep.missing_union_rules(root)), set(merge_prep.UNION_LOGS))
        (root / ".gitattributes").write_text("".join(f"{p} merge=union\n" for p in merge_prep.UNION_LOGS))
        self.assertEqual(merge_prep.missing_union_rules(root), [])

    def test_repo_gitattributes_declares_union_logs(self):
        self.assertEqual(merge_prep.missing_union_rules(REPO), [])


if __name__ == "__main__":
    unittest.main()
