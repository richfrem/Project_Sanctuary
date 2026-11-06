import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_gate as g


class Select(unittest.TestCase):
    def test_deterministic_first(self):
        self.assertEqual(g.select_method(True, True, True)["method"], "deterministic")
    def test_research_only_when_consequential_and_uncertain(self):
        self.assertEqual(g.select_method(False, True, True)["method"], "controlled_research")
        self.assertEqual(g.select_method(False, False, True)["method"], "bounded_model_review")
    def test_never_shrinks_preregistered(self):
        self.assertEqual(g.select_method(True, True, True, True)["method"], "checkpoint")


class Validity(unittest.TestCase):
    def test_qwen_regression_malformed_answers_block_behavioral_claim(self):
        # MISSION-004 confirm-v3: 58 of 60 independent-first Qwen answers were malformed.
        r = g.validity(60, 2, {"qwen_independent": {"k": 2, "n": 60}})
        self.assertEqual(r["max_claim"], "MECHANICS_ONLY")
        self.assertIn("ADHERENCE_LIMITED", r["problems"])
    def test_zero_baseline_unscorable(self):
        # MISSION-004 clean Haiku arm: 0/60 followed the planted majority at baseline.
        r = g.validity(60, 60, {"haiku_clean": {"k": 0, "n": 60}})
        self.assertEqual(r["max_claim"], "UNSCORABLE")
    def test_healthy_benchmark_allowed(self):
        r = g.validity(60, 58, {"base": {"k": 20, "n": 60}})
        self.assertEqual(r["max_claim"], "BEHAVIORAL_CLAIM_ALLOWED")
    def test_malformed_stay_in_denominator(self):
        self.assertEqual(g.validity(10, 5, {})["adherence"], 0.5)


class ValidityEdges(unittest.TestCase):
    def test_no_baseline_unscorable(self):
        self.assertEqual(g.validity(60, 60, {})["max_claim"], "UNSCORABLE")
    def test_bad_inputs_rejected(self):
        for args in ((0, 0, {}), (60, 70, {}), (-1, 0, {})):
            with self.assertRaises(ValueError): g.validity(*args)
        with self.assertRaises(ValueError): g.validity(60, 60, {"a": {"k": 1, "n": 2}}, min_adherence=0)


class Pilot(unittest.TestCase):
    R = [{"id": "a", "text": "ANSWER: 5"}, {"id": "b", "text": "I think five"}, {"id": "c", "text": "ANSWER: 7"}]
    def test_low_adherence_not_ready(self):
        self.assertFalse(g.pilot_adherence(self.R, r"^ANSWER: \d+$")["ready"])
    def test_pilot_items_reused_flagged(self):
        r = g.pilot_adherence(self.R[:1] + self.R[2:], r"ANSWER", confirmatory_ids=["c"])
        self.assertFalse(r["ready"]); self.assertTrue(any("reused" in p for p in r["problems"]))
    def test_ready(self):
        self.assertTrue(g.pilot_adherence(self.R[:1] + self.R[2:], r"ANSWER")["ready"])


class Fixtures(unittest.TestCase):
    def test_pass_and_fail(self):
        ok = {"name": "echo", "argv": [sys.executable, "-c", "print('hi')"], "expect_stdout_contains": ["hi"]}
        bad = {"name": "bad", "argv": [sys.executable, "-c", "raise SystemExit(3)"]}
        self.assertTrue(g.run_fixtures([ok])["passed"])
        self.assertFalse(g.run_fixtures([ok, bad])["passed"])


class Records(unittest.TestCase):
    def setUp(self):
        self.p = Path(tempfile.mkdtemp()) / "events.jsonl"
    def test_intervention_requires_observed_event_and_category(self):
        with self.assertRaises(ValueError): g.record_intervention("bogus", "r", "msg", path=self.p)
        with self.assertRaises(ValueError): g.record_intervention("scope_change", "r", "", path=self.p)
    def test_required_authorization_is_not_a_correction(self):
        a = g.record_intervention("required_authorization", "merge", "user said Go", path=self.p)
        b = g.record_intervention("execution_failure_correction", "stopped at partial", "user msg 3", path=self.p)
        self.assertFalse(a["counts_as_correction"]); self.assertTrue(b["counts_as_correction"])
    def test_repair_trigger_and_recurrence(self):
        for _ in range(2):
            g.record_repair("worker_env_preflight", "codex lacked claude on PATH", "", "", "mission-053", path=self.p)
        self.assertTrue(g.repair_report(self.p)["worker_env_preflight"]["repair_due"])
        g.record_repair("worker_env_preflight", "x", "exec check", "unit test", "mission-053", path=self.p)
        g.record_application("worker_env_preflight", "prevented", "u1", self.p)
        g.record_application("worker_env_preflight", "recurred", "u2", self.p)
        g.record_application("worker_env_preflight", "not_applicable", "u3", self.p)
        c = g.repair_report(self.p)["worker_env_preflight"]
        self.assertEqual((c["opportunities"], c["recurrence_rate"], c["repair_due"]), (2, 0.5, False))


class Invest(unittest.TestCase):
    def test_missing_is_unknown_not_zero(self):
        r = g.investment([{"id": "a", "execution": 100, "verification": 10, "orchestration_review": 20},
                          {"id": "b", "execution": 50}])
        self.assertTrue(r["telemetry_gap"]); self.assertIn("b", r["unknown_units"]["verification"])
        self.assertEqual(r["totals"]["execution"], 150); self.assertTrue(r["totals_are_lower_bounds"])
    def test_unmeasured_category_is_null_and_empty_is_gap(self):
        self.assertIsNone(g.investment([{"id": 1}])["totals"]["execution"])
        self.assertTrue(g.investment([])["telemetry_gap"])
    def test_repair_requires_evidence(self):
        with self.assertRaises(ValueError): g.record_repair("c", "", "r", "v", "o", path="/tmp/x.jsonl")


if __name__ == "__main__":
    unittest.main()
