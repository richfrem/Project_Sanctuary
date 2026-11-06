"""Sanctuary adapter over the shared evaluation skill: compatibility, separation of delivery from empirical state."""
import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_gate as g
import sanctuary_adapter as ad


class Adapter(unittest.TestCase):
    def setUp(self):
        self.m = {"id": "MISSION-999", "title": "Example", "status": "PROPOSED", "verdict": None,
                  "projected_impact": {"paths": ["LEARNING/experiments/mission-999/"]}}
        self.delivery = {"acceptance": [{"id": "A", "check": {"kind": "file_exists", "path": "x"}}],
                         "failure_conditions": [{"id": "F", "check": {"kind": "file_exists", "path": "bad"}}]}

    def test_shared_implementation_is_used(self):
        self.assertTrue((g.SHARED / "eval_core.py").is_file())
        self.assertIs(g.validity.__module__, "eval_core")        # no second implementation in Sanctuary

    def test_contract_defaults_human_gate_and_does_not_translate_hypothesis(self):
        c = ad.mission_to_contract(self.m, self.delivery)
        self.assertFalse(c["authorization"]["may_merge"]); self.assertEqual(c["type"], "sanctuary-mission")
        self.assertNotIn("hypothesis", json.dumps(c))

    def test_delivery_complete_does_not_create_verdict(self):
        p = Path(tempfile.mkdtemp()) / "events.jsonl"
        rec = ad.record_delivery(self.m, {"complete": True, "recommended": "accept", "incomplete": []}, p)
        self.assertTrue(rec["delivery_complete"]); self.assertEqual(rec["empirical_assessment"], "UNASSESSED")
        self.assertIsNone(self.m["verdict"])
        self.assertFalse(any(hasattr(ad, n) for n in ("record_result", "set_verdict", "corroborate")))

    def test_relative_store_path_not_doubled(self):
        import os
        d = Path(tempfile.mkdtemp()); old = os.getcwd(); os.chdir(d)
        try:
            ad.record_delivery(self.m, {"complete": True}, Path("./.agent/learning/events.jsonl"))
            self.assertTrue((d / ".agent/learning/events.jsonl").exists()); self.assertFalse((d / ".agent/learning/.agent").exists())
        finally:
            os.chdir(old)

    def test_qwen_shape_blocks_behavioral_verdict_via_shared_gate(self):
        v = g.validity(60, 2, {"q": {"k": 2, "n": 60}})                 # the 004 Qwen shape
        r = ad.interpret_validity(v, self.m)
        self.assertFalse(r["behavioral_verdict_supported"]); self.assertEqual(r["max_claim"], "MECHANICS_ONLY")
        ok = ad.interpret_validity(g.validity(60, 58, {"b": {"k": 20, "n": 60}}), self.m)
        self.assertTrue(ok["behavioral_verdict_supported"]); self.assertEqual(ok["current_empirical_state"], "UNASSESSED")

    def test_empirical_state_comes_only_from_manifest(self):
        self.assertEqual(ad.empirical_state(dict(self.m, verdict="FALSIFIED")), "FALSIFIED")

    def test_legacy_interface_and_event_type_preserved(self):
        p = Path(tempfile.mkdtemp()) / "e.jsonl"
        g.record_repair("c", "ev", "fix", "t", "o", path=p)
        self.assertEqual(json.loads(p.read_text())["event_type"], "FAILURE_REPAIR")
        self.assertEqual(g.repair_report(p)["c"]["failures"], 1)


if __name__ == "__main__":
    unittest.main()
