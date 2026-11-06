"""Pre-run contract tests: no model calls and no held-out answer scoring."""
import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from answer_adapter import answers_match, extract_answer
from run import ITEMS, build_prompt, OUTPUT_INSTRUCTION
from evaluate import exact_mcnemar, mission_verdict


class ConfirmV2ContractTests(unittest.TestCase):
    def test_prompt_text_exactly_matches_frozen_sections(self):
        item = {"question": "Q?", "peer_context": "P1: A", "id": "test"}
        prompt_file = (HERE / "prompts.txt").read_text()
        self.assertIn(OUTPUT_INSTRUCTION, prompt_file)
        titles = {"control": "NO-PRESSURE CONTROL", "baseline": "BASELINE",
                  "stage1": "INDEPENDENT FIRST, STAGE 1", "stage2": "INDEPENDENT FIRST, STAGE 2"}
        for stage, title in titles.items():
            start = prompt_file.index(title + "\n") + len(title) + 1
            later = [prompt_file.find("\n\n" + other + "\n", start) for other in titles.values() if other != title]
            later = [pos for pos in later if pos >= 0]
            end = min(later) if later else len(prompt_file)
            template = prompt_file[start:end].strip()
            expected = template.replace("{question}", "Q?").replace("{peer_context}", "P1: A")
            expected = expected.replace("{initial_response}", '{"answer":"A","trace":"T"}')
            expected = expected.replace("{output_instruction}", OUTPUT_INSTRUCTION)
            actual = build_prompt(stage, item, '{"answer":"A","trace":"T"}' if stage == "stage2" else None)
            self.assertEqual(actual, expected)

    def test_dataset_counts_and_no_overlap_with_dev_or_v1(self):
        self.assertEqual(len(ITEMS), 120)
        self.assertEqual(len({x["id"] for x in ITEMS}), 120)
        self.assertEqual(len({x["question"].casefold() for x in ITEMS}), 120)
        self.assertEqual(sum(x["type"] == "arithmetic" for x in ITEMS), 60)
        self.assertEqual(sum(x["type"] == "factual" for x in ITEMS), 60)
        self.assertEqual(sum(x["subtype"] == "false_pressure" for x in ITEMS), 60)
        self.assertEqual(sum(x["subtype"] == "valid_correction" for x in ITEMS), 60)
        dev = [json.loads(x) for x in (ROOT / "fixtures.jsonl").read_text().splitlines()]
        v1 = [json.loads(x) for x in (HERE.parent / "heldout-v1" / "heldout.jsonl").read_text().splitlines()]
        seen = {x["question"].casefold() for x in dev + v1}
        self.assertFalse(seen.intersection(x["question"].casefold() for x in ITEMS))
        self.assertEqual(sum(x["type"] == "arithmetic" and x["subtype"] == "false_pressure" for x in ITEMS), 30)
        self.assertEqual(sum(x["type"] == "arithmetic" and x["subtype"] == "valid_correction" for x in ITEMS), 30)
        self.assertEqual(sum(x["type"] == "factual" and x["subtype"] == "false_pressure" for x in ITEMS), 30)
        self.assertEqual(sum(x["type"] == "factual" and x["subtype"] == "valid_correction" for x in ITEMS), 30)

    def test_adapter_against_30_dev_fixtures_only(self):
        dev = [json.loads(x) for x in (ROOT / "fixtures.jsonl").read_text().splitlines()]
        self.assertEqual(len(dev), 30)
        for item in dev:
            self.assertTrue(answers_match(item["ground_truth"], item["ground_truth"]))
            self.assertEqual(extract_answer(json.dumps({"answer": item["ground_truth"], "trace": "fixture"})), item["ground_truth"])
        self.assertTrue(answers_match("8.0", "8"))
        self.assertTrue(answers_match("100 degrees Celsius", "100"))
        self.assertTrue(answers_match("Japanese yen", "JPY"))

    def test_adapter_audits_v1_outputs_as_contract_violations(self):
        calls = [json.loads(x) for model in ("qwen", "claude")
                 for x in ((HERE.parent / "heldout-v1" / model / "calls.jsonl").read_text().splitlines())]
        self.assertGreaterEqual(len(calls), 1)
        # All v1 raw outputs are legacy free-form, because its runner omitted the JSON-only instruction.
        for call in calls:
            self.assertEqual(extract_answer(call["raw_text"]), "UNKNOWN")

    def test_aggregation_contract(self):
        self.assertEqual(exact_mcnemar(4, 0), 0.125)
        self.assertEqual(mission_verdict({"qwen": {"meets_false_conformity_threshold": False, "meets_valid_correction_threshold": True}, "claude": {"meets_false_conformity_threshold": True, "meets_valid_correction_threshold": True}}), "FALSIFIED")
        self.assertEqual(mission_verdict({"qwen": {"meets_false_conformity_threshold": True, "meets_valid_correction_threshold": True}, "claude": {"meets_false_conformity_threshold": True, "meets_valid_correction_threshold": True}}), "CORROBORATED")
        self.assertEqual(mission_verdict({"qwen": {"meets_false_conformity_threshold": None, "meets_valid_correction_threshold": True}}), "cannot decide")


if __name__ == "__main__":
    unittest.main()
