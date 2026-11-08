"""Deterministic pre-freeze regression checks; no model calls or held-out data."""

import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import analyze
import choose
import common
import run_study
import worker_containment
import freeze_study
import scenarios


class InstrumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_unknown_usage_is_not_zero(self):
        rows = [dict(call_id="known", usage=dict(additional_tokens=11)),
                dict(call_id="missing", usage=None),
                dict(call_id="invalid", usage=dict(additional_tokens=-1))]
        self.assertEqual(common.usage_totals(rows), (11, ["missing", "invalid"]))

    def test_codex_reply_excludes_echoed_input_and_diagnostics(self):
        path = self.root / "reply.jsonl"
        rows = [dict(type="response_item", payload=dict(type="message", role="user", content=[dict(type="input_text", text='{"echo":"wrong"}')])),
                dict(type="response_item", payload=dict(type="message", role="assistant", phase="final_answer", content=[dict(type="output_text", text='{"answer":"right"}')]))]
        path.write_text("\n".join(map(json.dumps, rows)))
        self.assertEqual(common.extract_json(common.codex_final_text(path)), dict(answer="right"))

    def test_claude_reply_uses_assistant_text_not_warning_output(self):
        path = self.root / "reply.jsonl"
        rows = [dict(type="system", message="warning"),
                dict(type="assistant", message=dict(content=[dict(type="thinking", thinking="private"),
                                                             dict(type="text", text="METERING_OK")]))]
        path.write_text("\n".join(map(json.dumps, rows)))
        self.assertEqual(common.claude_final_text(path), "METERING_OK")

    def test_sandbox_profile_write_does_not_follow_worker_planted_link(self):
        root = self.root / "worker"
        root.mkdir()
        profile, runtime = worker_containment.prepare(root)
        outside = self.root / "outside.txt"
        outside.write_text("preserve")
        profile.unlink()
        profile.symlink_to(outside)
        worker_containment.prepare(root)
        self.assertEqual(outside.read_text(), "preserve")
        self.assertFalse(profile.is_symlink())

    def test_sandbox_refuses_symlinked_deliverables(self):
        root = self.root / "worker"
        root.mkdir()
        (root / "deliverables").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            worker_containment.prepare(root)

    def test_attribution_requires_unique_marker_and_fresh_session(self):
        old = self.root / "old.jsonl"
        new = self.root / "new.jsonl"
        other = self.root / "other.jsonl"
        old.write_text("tag")
        new.write_text("tag")
        other.write_text("unrelated")
        with patch.object(common, "transcript_files", return_value=[str(old), str(new), str(other)]):
            self.assertEqual(common.attributed_transcript("agy", 0, {str(old)}, "tag"), str(new))
            self.assertIsNone(common.attributed_transcript("agy", 0, marker="tag"))
            self.assertIsNone(common.attributed_transcript("agy", 0, marker="absent"))
            self.assertIsNone(common.attributed_transcript("agy", 0))

    def test_agy_measurement_from_attributed_fixture(self):
        path = self.root / "agy.jsonl"
        rows = [dict(tag="unique"),
                dict(type="PLANNER_RESPONSE", input_tokens=100, cache_read_tokens=20, output_tokens=7),
                dict(type="PLANNER_RESPONSE", input_tokens=150, cache_read_tokens=30, output_tokens=9)]
        path.write_text("\n".join(map(json.dumps, rows)))
        with patch.object(common, "transcript_files", return_value=[str(path)]):
            result = common.measure_agy(0, marker="unique")
        self.assertEqual(result["additional_tokens"], 76)
        self.assertEqual(result["accumulated_input"], 300)

    def test_codex_duplicate_usage_events_are_not_double_counted(self):
        path = self.root / "codex.jsonl"
        event = dict(payload=dict(type="token_count", info=dict(
            total_token_usage=dict(total_tokens=105),
            last_token_usage=dict(input_tokens=100, output_tokens=5))))
        path.write_text(json.dumps(dict(tag="unique")) + "\n" + json.dumps(event) + "\n" + json.dumps(event))
        with patch.object(common, "transcript_files", return_value=[str(path)]):
            result = common.measure_codex(0, marker="unique")
        self.assertEqual(result["additional_tokens"], 5)
        self.assertEqual(result["requests"], 1)

    def test_heartbeat_renews_codex_and_propagates_failure(self):
        with patch.object(run_study.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "heartbeat")) as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                run_study.heartbeat()
        args = execute.call_args.args[0]
        self.assertEqual(args[args.index("--agent") + 1], "codex")
        self.assertTrue(execute.call_args.kwargs["check"])

    def test_unknown_quota_stops(self):
        response = subprocess.CompletedProcess([], 0, "agy status=ok tightest_window_remaining=80%\n", "")
        with patch.object(run_study.subprocess, "run", return_value=response):
            with self.assertRaises(SystemExit):
                run_study.headroom()

    def test_any_low_backend_stops_flash(self):
        with patch.object(run_study, "headroom", return_value=dict(claude=50, agy=90, codex=18)):
            with self.assertRaises(SystemExit):
                run_study.check_stop("flash")

    def test_unknown_usage_prevents_next_dispatch(self):
        self.write("usage-log.jsonl", dict(call_id="missing", usage=None))
        with patch.object(run_study, "STUDY", self.root), patch.object(run_study, "check_stop"), patch.object(run_study, "run_model") as model:
            with self.assertRaises(SystemExit):
                run_study.call("flash", "input", "instruction", "calibration", "next")
        model.assert_not_called()

    def test_existing_call_id_cannot_be_dispatched_again(self):
        self.write("usage-log.jsonl", dict(call_id="already-used", usage=None))
        with patch.object(common, "STUDY", self.root), patch.object(common.subprocess, "run") as model:
            with self.assertRaises(ValueError):
                common.run_model("flash", "input", "instruction", "calibration", "already-used",
                                 cwd=self.root, out_dir=self.root / "raw")
        model.assert_not_called()

    def test_string_calibration_judgment_is_rejected(self):
        checks = {key: dict(value="false") for key in ("D1", "D2", "D3")}
        with patch.object(run_study, "STUDY", self.root), patch.object(run_study, "neutral_check"), patch.object(run_study, "call", return_value=dict(text=json.dumps(dict(checks=checks)))):
            with self.assertRaises(ValueError):
                run_study.cmd_calibrate()

    def test_budget_ceiling_prevents_next_dispatch(self):
        self.write("usage-log.jsonl", dict(call_id="spent", phase="calibration",
                   usage=dict(additional_tokens=run_study.T["total_additional_tokens_ceiling"])))
        with patch.object(run_study, "STUDY", self.root):
            with self.assertRaises(SystemExit):
                run_study.check_budget()

    def test_worker_overrun_stops(self):
        self.write("usage-log.jsonl", dict(call_id="work-ITEM-01", phase="execution",
                   usage=dict(additional_tokens=run_study.T["per_item_cap_additional_tokens"] + 1)))
        with patch.object(run_study, "STUDY", self.root):
            with self.assertRaises(SystemExit):
                run_study.check_budget()

    def test_flash_luna_calibration_uses_nine_boolean_judgments(self):
        checks = {key: dict(value=True) for key in ("D1", "D2", "D3")}
        with patch.object(run_study, "STUDY", self.root), patch.object(run_study, "neutral_check"), patch.object(run_study, "call", return_value=dict(text=json.dumps(dict(checks=checks)))) as model:
            with contextlib.redirect_stdout(io.StringIO()):
                run_study.cmd_calibrate()
        result = json.loads((self.root / "calibration-results.json").read_text())
        self.assertEqual(result["raters"], ["flash", "luna"])
        self.assertEqual(result["agreement"], 1)
        self.assertEqual(result["total"], 9)
        self.assertEqual(model.call_count, 6)

    def test_single_worker_selection_preserves_disjoint_cohorts(self):
        cards = [dict(id=f"DEV-{i:02}", title="dev", summary="fixture") for i in range(22)]
        self.write("pool.json", cards)
        self.write("formula.json", {c["id"]: dict(strategic_leverage=i + 1, epistemic_depth=1,
                   cost_weight=1, operational_friction=1) for i, c in enumerate(cards)})
        for rater in choose.T["declarers"]:
            self.write(f"declarations/{rater}.json", [dict(id=c["id"], explanation="fixture", **{d: 3 for d in choose.T["dimensions"]}) for c in cards])
        with patch.object(choose, "STUDY", self.root), patch.object(choose, "verify_freeze"), patch.object(choose, "kb_docs", return_value={}), patch.object(choose, "containment", return_value=0):
            with contextlib.redirect_stdout(io.StringIO()):
                choose.main()
        result = json.loads((self.root / "selection.json").read_text())
        self.assertEqual(set(result["workers"].values()), {"flash"})
        self.assertEqual(set(result["reviewer"].values()), {"luna"})
        self.assertEqual(set(result["successor"].values()), {"flash"})
        self.assertEqual(len(result["item_codes"]), 10)
        self.assertFalse(set(result["P_only"]) & set(result["F_only"]))

    def test_unknown_nonworker_usage_cannot_yield_verdict(self):
        self.write("selection.json", dict(cohort_of={}))
        self.write("usage-log.jsonl", dict(call_id="calibration", usage=None))
        with patch.object(analyze, "STUDY", self.root):
            with contextlib.redirect_stdout(io.StringIO()):
                analyze.main()
        result = json.loads((self.root / "results.json").read_text())
        self.assertEqual(result["verdict"], "UNASSESSED")
        self.assertIsNone(result["claim_B"]["B2"]["total_additional_tokens"])
        self.assertFalse(result["validity_gates"]["usage_measured"])

    def test_item_cap_includes_successor_and_reviewer(self):
        rows = [dict(call_id=f"{role}-ITEM-01", phase=phase, usage=dict(additional_tokens=value))
                for role, phase, value in (("work", "execution", 15000), ("succ", "successor", 6000), ("review", "review", 5000))]
        (self.root / "usage-log.jsonl").write_text("\n".join(map(json.dumps, rows)))
        self.assertEqual(common.item_usage(rows, "ITEM-01")["additional_tokens"], 26000)
        with patch.object(run_study, "STUDY", self.root):
            with self.assertRaises(SystemExit):
                run_study.check_budget()

    def test_missing_item_call_is_not_complete_usage(self):
        row = dict(call_id="work-ITEM-01", usage=dict(additional_tokens=20))
        self.assertFalse(common.item_usage([row], "ITEM-01")["calls_complete"])
        self.assertTrue(common.item_usage([row], "ITEM-01", expected={"work-ITEM-01"})["calls_complete"])

    def test_declaration_rejects_missing_duplicate_and_noninteger_ratings(self):
        pool = [dict(id="A"), dict(id="B", proposed_by="flash")]
        row = dict(id="A", explanation="fixture", **{key: 3 for key in choose.T["dimensions"]})
        common.validate_declaration([row], pool, "flash", choose.T["dimensions"])
        for bad in ([], [row, row], [dict(row, curiosity="3")], [dict(row, curiosity=True)]):
            with self.assertRaises(ValueError):
                common.validate_declaration(bad, pool, "flash", choose.T["dimensions"])

    def analysis_fixture(self, claim_a="MET"):
        cohorts, log = {}, []
        for index in range(8):
            code, identity = f"ITEM-{index + 1:02}", f"DEV-{index}"
            cohorts[identity] = "P-only" if index < 4 else "F-only"
            checks = {key: dict(value=True) for key in ("D1", "D2", "D3", "S1", "S2", "S3", "S4", "S5")}
            checks["D1"]["value"] = index < 6
            self.write(f"assess/{code}.json", dict(code=code, id=identity, kind="discovery", worker="flash",
                       review=dict(checks=checks), deterministic=dict(files=["fixture"], findings_present=True), worker_status="DONE"))
            log.extend(dict(call_id=f"{role}-{code}", phase=phase, usage=dict(additional_tokens=10))
                       for role, phase in (("work", "execution"), ("succ", "successor"), ("review", "review")))
        self.write("selection.json", dict(cohort_of=cohorts))
        self.write("calibration-results.json", dict(agreement=1))
        self.write("declarations/attestation.json", dict(blind=True))
        self.write("scenario-results-heldout.json", dict(claim_A=claim_a))
        (self.root / "usage-log.jsonl").write_text("\n".join(map(json.dumps, log)))

    def run_analysis(self):
        with patch.object(analyze, "STUDY", self.root), contextlib.redirect_stdout(io.StringIO()):
            analyze.main()
        return json.loads((self.root / "results.json").read_text())

    def test_primary_denominator_counts_all_item_calls(self):
        self.analysis_fixture()
        result = self.run_analysis()
        self.assertEqual(result["cohorts"]["P_only"]["tokens"], 120)
        self.assertEqual(result["verdict"], "CORROBORATED")

    def test_unrecognized_claim_a_cannot_corroborate(self):
        for value in ("unexpected data", None):
            self.analysis_fixture(value)
            self.assertEqual(self.run_analysis()["verdict"], "UNASSESSED")

    def test_string_false_review_makes_evidence_unassessed(self):
        self.analysis_fixture()
        path = self.root / "assess/ITEM-01.json"
        data = json.loads(path.read_text())
        data["review"]["checks"]["D1"]["value"] = "false"
        path.write_text(json.dumps(data))
        result = self.run_analysis()
        self.assertEqual(result["verdict"], "UNASSESSED")
        self.assertFalse(result["validity_gates"]["review_contract"])

    def test_missing_review_structure_does_not_crash_analysis(self):
        self.analysis_fixture()
        path = self.root / "assess/ITEM-01.json"
        data = json.loads(path.read_text())
        data["review"] = {}
        path.write_text(json.dumps(data))
        self.assertEqual(self.run_analysis()["verdict"], "UNASSESSED")

    def test_unassessed_worker_attempt_cannot_be_silently_dropped(self):
        self.analysis_fixture()
        with (self.root / "usage-log.jsonl").open("a") as stream:
            stream.write("\n" + json.dumps(dict(call_id="work-ITEM-09", phase="execution", usage=dict(additional_tokens=100))))
        result = self.run_analysis()
        self.assertEqual(result["verdict"], "UNASSESSED")
        self.assertFalse(result["validity_gates"]["attempt_records_complete"])

    def freeze_fixture(self):
        for name in freeze_study.STATIC:
            (self.root / name).write_text("fixture")
        self.write("LEARNING/missions/MISSION-060.json", dict(id="MISSION-060"))
        (self.root / "usage-log.jsonl").write_text("original\n")
        (self.root / "instrument.py").write_text("# fixture\n")

    def test_freeze_allows_usage_append_but_detects_source_mutation(self):
        self.freeze_fixture()
        with patch.object(freeze_study, "STUDY", self.root), patch.object(freeze_study, "REPO", self.root), patch.object(freeze_study, "dependency_paths", return_value={}), patch.object(freeze_study, "cli_versions", return_value={}):
            freeze_study.create_freeze()
            freeze_study.verify_freeze()
            (self.root / "usage-log.jsonl").write_text("original\nnew-call\n")
            freeze_study.verify_freeze()
            (self.root / "instrument.py").write_text("# changed\n")
            with self.assertRaises(ValueError):
                freeze_study.verify_freeze()

    def test_freeze_detects_prefix_mutation_and_requires_integration(self):
        self.freeze_fixture()
        with patch.object(freeze_study, "STUDY", self.root), patch.object(freeze_study, "REPO", self.root), patch.object(freeze_study, "dependency_paths", return_value={}), patch.object(freeze_study, "cli_versions", return_value={}):
            freeze_study.create_freeze()
            with patch.object(freeze_study.subprocess, "run", side_effect=subprocess.CalledProcessError(128, "git show")):
                with self.assertRaises(subprocess.CalledProcessError):
                    freeze_study.verify_freeze(require_integrated=True)
            (self.root / "usage-log.jsonl").write_text("changed\n")
            with self.assertRaises(ValueError):
                freeze_study.verify_freeze()

    def test_heldout_guard_refuses_existing_result_without_evaluation(self):
        self.write("scenario-results-heldout.json", dict(fixture=True))
        with patch.object(scenarios, "STUDY", self.root), patch.object(freeze_study, "verify_freeze"), patch.object(scenarios, "evaluate") as evaluate:
            with self.assertRaises(ValueError):
                scenarios.main("heldout")
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
