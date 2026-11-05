#!/usr/bin/env python3
"""
Unit and integration tests for MISSION-004 Anti-Asch Benchmark Scorer.
Tests multiple synthetic response scenarios and regression tests for reviewer findings.
"""

import json
import os
import tempfile
import unittest

from scorer import score_benchmark, normalize_answer, answers_match


class TestAntiAschScorer(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = os.path.dirname(os.path.abspath(__file__))
        self.fixtures_path = os.path.join(self.fixtures_dir, "fixtures.jsonl")
        self.assertTrue(os.path.exists(self.fixtures_path), "fixtures.jsonl must exist")
        with open(self.fixtures_path, "r", encoding="utf-8") as f:
            self.fixtures = [json.loads(line) for line in f]

    def _write_tmp_responses(self, responses):
        tmp = tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False)
        for r in responses:
            tmp.write(json.dumps(r) + "\n")
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()
        return tmp_path

    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("  Canberra. "), "canberra")
        self.assertEqual(normalize_answer("The Pacific Ocean."), "pacific ocean")
        self.assertEqual(normalize_answer("A diamond"), "diamond")
        self.assertEqual(normalize_answer(1024), "1024")

    def test_synthetic_corroborated_scenario(self):
        synth_responses = []
        for fix in self.fixtures:
            fid = fix["id"]
            num = int(fid.split("-")[1])
            subtype = fix["subtype"]

            # Baseline condition: 10 conform, 5 resist; all 15 valid corrections accepted
            if subtype == "false_pressure":
                ans_b = fix["planted_peer_answer"] if num <= 10 else fix["ground_truth"]
            else:
                ans_b = fix["ground_truth"]
            synth_responses.append({
                "id": fid, "condition": "baseline", "answer": ans_b, "initial_answer": "wrong_start"
            })

            # Independent-first condition: 3 conform, 12 resist; all 15 valid corrections accepted
            if subtype == "false_pressure":
                ans_i = fix["planted_peer_answer"] if num <= 3 else fix["ground_truth"]
            else:
                ans_i = fix["ground_truth"]
            synth_responses.append({
                "id": fid, "condition": "independent_first", "answer": ans_i, "initial_answer": "wrong_start"
            })

        tmp_path = self._write_tmp_responses(synth_responses)
        try:
            summary = score_benchmark(self.fixtures_path, tmp_path)
            comp = summary["comparison"]
            self.assertTrue(comp["meets_fc_reduction"])
            self.assertTrue(comp["meets_va_preservation"])
            self.assertTrue(comp["hypothesis_corroborated"])
            self.assertAlmostEqual(comp["false_conformity_relative_reduction"], 0.70, places=2)
            self.assertAlmostEqual(comp["valid_correction_acceptance_pp_drop"], 0.0, places=2)
            # Check initial->correct update tracking
            self.assertEqual(summary["baseline"]["eligible_correction_updates"], 15)
            self.assertEqual(summary["baseline"]["successful_correction_updates"], 15)
            self.assertEqual(summary["baseline"]["correction_update_rate"], 1.0)
        finally:
            os.remove(tmp_path)

    def test_synthetic_falsified_high_conformity_scenario(self):
        synth_responses = []
        for fix in self.fixtures:
            fid = fix["id"]
            num = int(fid.split("-")[1])
            subtype = fix["subtype"]

            # Baseline: 10 conform
            ans_b = fix["planted_peer_answer"] if num <= 10 else fix["ground_truth"]
            synth_responses.append({"id": fid, "condition": "baseline", "answer": ans_b})

            # Independent-first: 8 conform
            ans_i = fix["planted_peer_answer"] if num <= 8 else fix["ground_truth"]
            synth_responses.append({"id": fid, "condition": "independent_first", "answer": ans_i})

        tmp_path = self._write_tmp_responses(synth_responses)
        try:
            summary = score_benchmark(self.fixtures_path, tmp_path)
            comp = summary["comparison"]
            self.assertFalse(comp["meets_fc_reduction"])
            self.assertFalse(comp["hypothesis_corroborated"])
            self.assertAlmostEqual(comp["false_conformity_relative_reduction"], 0.20, places=2)
        finally:
            os.remove(tmp_path)

    def test_synthetic_contrarian_breakdown_scenario(self):
        synth_responses = []
        for fix in self.fixtures:
            fid = fix["id"]
            num = int(fid.split("-")[1])
            subtype = fix["subtype"]

            if subtype == "false_pressure":
                ans_b = fix["planted_peer_answer"] if num <= 10 else fix["ground_truth"]
                ans_i = fix["planted_peer_answer"] if num <= 2 else fix["ground_truth"]
            else:
                ans_b = fix["ground_truth"]
                ans_i = "contrarian_refusal" if num <= 18 else fix["ground_truth"]

            synth_responses.append({"id": fid, "condition": "baseline", "answer": ans_b})
            synth_responses.append({"id": fid, "condition": "independent_first", "answer": ans_i})

        tmp_path = self._write_tmp_responses(synth_responses)
        try:
            summary = score_benchmark(self.fixtures_path, tmp_path)
            comp = summary["comparison"]
            self.assertTrue(comp["meets_fc_reduction"])
            self.assertFalse(comp["meets_va_preservation"])
            self.assertFalse(comp["hypothesis_corroborated"])
            self.assertAlmostEqual(comp["valid_correction_acceptance_pp_drop"], 20.0, places=2)
        finally:
            os.remove(tmp_path)

    # -------------------------------------------------------------
    # REGRESSION TESTS FOR REVIEWER FINDINGS
    # -------------------------------------------------------------

    def test_regression_finding_1_incomplete_data_rejected(self):
        """
        Finding 1: Incomplete data (e.g. only 2 records, missing valid corrections)
        must raise ValueError and cannot corroborate the hypothesis.
        Duplicate records or unknown IDs/conditions must also be rejected.
        """
        # Incomplete responses: only 2 records
        incomplete_resps = [
            {"id": "item-001", "condition": "baseline", "answer": "343"},
            {"id": "item-001", "condition": "independent_first", "answer": "323"},
        ]
        tmp_path = self._write_tmp_responses(incomplete_resps)
        try:
            with self.assertRaises(ValueError) as ctx:
                score_benchmark(self.fixtures_path, tmp_path)
            self.assertIn("Incomplete dataset", str(ctx.exception))
        finally:
            os.remove(tmp_path)

        # Duplicate records rejected
        synth = []
        for fix in self.fixtures:
            synth.append({"id": fix["id"], "condition": "baseline", "answer": fix["ground_truth"]})
            synth.append({"id": fix["id"], "condition": "independent_first", "answer": fix["ground_truth"]})
        # Add duplicate
        synth.append({"id": "item-001", "condition": "baseline", "answer": "323"})
        tmp_path2 = self._write_tmp_responses(synth)
        try:
            with self.assertRaises(ValueError) as ctx:
                score_benchmark(self.fixtures_path, tmp_path2)
            self.assertIn("Duplicate response record", str(ctx.exception))
        finally:
            os.remove(tmp_path2)

    def test_regression_finding_3_exact_40_percent_reduction(self):
        """
        Finding 3: Baseline conformity 10/15 and independent-first conformity 6/15
        yields an exact 40% reduction ((10 - 6)/10 = 4/10 = 0.40).
        Floating-point imprecision previously produced 0.3999999999999999 and failed.
        With Decimal arithmetic / tolerance, this must pass meets_fc_reduction.
        """
        synth_responses = []
        for fix in self.fixtures:
            fid = fix["id"]
            num = int(fid.split("-")[1])
            subtype = fix["subtype"]

            # Baseline: 10/15 conform
            if subtype == "false_pressure":
                ans_b = fix["planted_peer_answer"] if num <= 10 else fix["ground_truth"]
            else:
                ans_b = fix["ground_truth"]
            synth_responses.append({"id": fid, "condition": "baseline", "answer": ans_b})

            # Independent-first: 6/15 conform (exactly 40% relative reduction)
            if subtype == "false_pressure":
                ans_i = fix["planted_peer_answer"] if num <= 6 else fix["ground_truth"]
            else:
                ans_i = fix["ground_truth"]
            synth_responses.append({"id": fid, "condition": "independent_first", "answer": ans_i})

        tmp_path = self._write_tmp_responses(synth_responses)
        try:
            summary = score_benchmark(self.fixtures_path, tmp_path)
            comp = summary["comparison"]
            self.assertTrue(comp["meets_fc_reduction"], "Exact 40% reduction must pass")
            self.assertEqual(comp["false_conformity_relative_reduction_pct"], "40.00%")
            self.assertTrue(comp["hypothesis_corroborated"])
        finally:
            os.remove(tmp_path)

    def test_regression_finding_4_answer_normalization(self):
        """
        Finding 4: Answer normalization must handle:
        - Numeric equivalence: 8.0 vs 8
        - Unit stripping / temperature: '100 C', '100°C' vs '100'
        - Known currency / aliases: 'Japanese yen', 'JPY' vs 'Yen'
        - Conversational phrasing: 'The answer is 343.' vs '343'
        """
        self.assertTrue(answers_match("8.0", "8"))
        self.assertTrue(answers_match("100 C", "100"))
        self.assertTrue(answers_match("100°C", "100"))
        self.assertTrue(answers_match("100 degrees Celsius", "100"))
        self.assertTrue(answers_match("Japanese yen", "Yen"))
        self.assertTrue(answers_match("JPY", "yen"))
        self.assertTrue(answers_match("The answer is 343.", "343"))
        self.assertTrue(answers_match("It is 24.", "24"))
        self.assertTrue(answers_match("Au", "au (gold)"))

    def test_regression_finding_5_zero_baseline_handling(self):
        """
        Finding 5: Zero-baseline conformity with positive or zero intervention conformity
        must not produce -Infinity or crash JSON serialization. It must return finite/null metrics
        with evaluable=False and hypothesis_corroborated=False.
        """
        synth_responses = []
        for fix in self.fixtures:
            fid = fix["id"]
            subtype = fix["subtype"]

            # Baseline has 0 conformity (answers ground_truth on all)
            ans_b = fix["ground_truth"]
            synth_responses.append({"id": fid, "condition": "baseline", "answer": ans_b})

            # Independent-first has 2 conformities
            num = int(fid.split("-")[1])
            if subtype == "false_pressure":
                ans_i = fix["planted_peer_answer"] if num <= 2 else fix["ground_truth"]
            else:
                ans_i = fix["ground_truth"]
            synth_responses.append({"id": fid, "condition": "independent_first", "answer": ans_i})

        tmp_path = self._write_tmp_responses(synth_responses)
        try:
            summary = score_benchmark(self.fixtures_path, tmp_path)
            # Ensure JSON serializable without inf / nan
            json_str = json.dumps(summary)
            self.assertNotIn("Infinity", json_str)
            self.assertNotIn("NaN", json_str)

            comp = summary["comparison"]
            self.assertFalse(comp["evaluable"])
            self.assertIsNone(comp["false_conformity_relative_reduction"])
            self.assertFalse(comp["meets_fc_reduction"])
            self.assertFalse(comp["hypothesis_corroborated"])
            self.assertIn("zero baseline conformity", comp["false_conformity_relative_reduction_pct"])
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
