"""Tests for usage_probe.py (MISSION-035). Fixtures are real captured outputs (tests/fixtures)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import usage_probe as up  # noqa: E402

FIX = HERE / "fixtures"


class TestParsers(unittest.TestCase):
    def test_parse_claude_usage_real_output(self):
        r = up.parse_claude_usage((FIX / "claude_usage.txt").read_text())
        self.assertIsInstance(r["session_pct_used"], int)
        self.assertIsInstance(r["week_pct_used"], int)
        self.assertIn("Nov", r["session_resets"])
        self.assertIn("Nov", r["week_resets"])

    def test_parse_claude_usage_missing_returns_none_fields(self):
        r = up.parse_claude_usage("garbage output")
        self.assertIsNone(r["session_pct_used"])
        self.assertIsNone(r["week_pct_used"])

    def test_parse_agy_usage_real_output(self):
        rows = up.parse_agy_usage(json.loads((FIX / "agy_usage.json").read_text()))
        self.assertGreaterEqual(len(rows), 5)
        for row in rows:
            self.assertIn(row["window"], ("weekly", "5h", "other"))
            self.assertTrue(0.0 <= row["remaining_pct"] <= 100.0)
            self.assertTrue(row["resets_at"])
        groups = {r["group"] for r in rows}
        self.assertIn("Gemini Models", groups)

    def test_agy_cost_is_zero_tokens(self):
        d = json.loads((FIX / "agy_usage.json").read_text())
        self.assertEqual(up.reported_tokens("agy", d), 0)


def _rollout(path, events):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _tc(ts, p5, p7, total):
    return {"timestamp": ts, "type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": total, "output_tokens": 1, "total_tokens": total + 1}},
        "rate_limits": {"primary": {"used_percent": p5, "window_minutes": 300, "resets_at": 1793820154},
                        "secondary": {"used_percent": p7, "window_minutes": 10080, "resets_at": 1794406954}}}}


class TestCodex(unittest.TestCase):
    def test_scans_back_past_files_without_token_count(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "2026" / "11" / "04").mkdir(parents=True)
            _rollout(d / "2026/11/04/rollout-a.jsonl", [_tc("2026-11-04T10:00:00Z", 10.0, 50.0, 100),
                                                         _tc("2026-11-04T10:05:00Z", 12.0, 51.0, 900)])
            _rollout(d / "2026/11/04/rollout-b.jsonl", [{"type": "session_meta", "payload": {}}])
            r = up.read_codex_latest(d)
            self.assertEqual(r["five_hour_pct_used"], 12.0)
            self.assertEqual(r["weekly_pct_used"], 51.0)
            self.assertEqual(r["session_total_tokens"], 901)
            self.assertEqual(r["source_timestamp"], "2026-11-04T10:05:00Z")

    def test_no_data_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(up.read_codex_latest(Path(d)))

    def test_never_opens_auth_json(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "auth.json").write_text('{"tokens": {"access_token": "sk-secret-123456789012345678"}}')
            up.read_codex_latest(d)
            snap = json.dumps(up.read_codex_latest(d))
            self.assertNotIn("sk-secret", snap)


def _tx(path, rows):
    with open(path, "w") as f:
        for ts, i, o in rows:
            f.write(json.dumps({"type": "assistant", "timestamp": ts, "message": {"role": "assistant",
                    "usage": {"input_tokens": i, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0, "output_tokens": o}}}) + "\n")


class TestClaudeTranscripts(unittest.TestCase):
    def test_sums_tokens_in_interval_only(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _tx(d / "s1.jsonl", [("2026-11-04T10:00:00Z", 10, 5), ("2026-11-04T10:10:00Z", 100, 50),
                                 ("2026-11-04T10:20:00Z", 7, 3)])
            r = up.claude_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z")
            self.assertEqual(r["input_tokens"], 100)
            self.assertEqual(r["output_tokens"], 50)
            self.assertFalse(r["concurrent"])

    def test_concurrent_flag_when_second_transcript_active(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _tx(d / "s1.jsonl", [("2026-11-04T10:10:00Z", 100, 50)])
            _tx(d / "s2.jsonl", [("2026-11-04T10:11:00Z", 9, 9)])
            r = up.claude_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z", own_session="s1")
            self.assertTrue(r["concurrent"])
            self.assertEqual(r["input_tokens"], 100)

    def test_own_session_alone_not_concurrent(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _tx(d / "s1.jsonl", [("2026-11-04T10:10:00Z", 100, 50)])
            r = up.claude_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z", own_session="s1")
            self.assertFalse(r["concurrent"])


class TestSnapshotDiffAndSecrets(unittest.TestCase):
    def _runner(self, agy_remaining=0.5, claude_pct=2):
        agy = json.loads((FIX / "agy_usage.json").read_text())
        for g in agy["command"]["data"]["groups"]:
            for b in g["buckets"]:
                if b["id"] == "gemini-weekly":
                    b["remaining_fraction"] = agy_remaining
        claude = (FIX / "claude_usage.txt").read_text().replace("Current session: 9%", f"Current session: {claude_pct}%")

        def run(argv, timeout=60):
            if argv[0] == "claude":
                return claude
            if argv[0] == "agy":
                return json.dumps(agy)
            raise AssertionError(argv)
        return run

    def test_snapshot_collects_available_backends(self):
        s = up.snapshot(["claude", "agy"], runner=self._runner(), now="2026-11-04T10:00:00Z")
        self.assertEqual(s["timestamp"], "2026-11-04T10:00:00Z")
        self.assertIn("claude", s["backends"])
        self.assertIn("agy", s["backends"])

    def test_diff_window_deltas(self):
        a = up.snapshot(["claude", "agy"], runner=self._runner(0.5, 2), now="2026-11-04T10:00:00Z")
        b = up.snapshot(["claude", "agy"], runner=self._runner(0.4, 5), now="2026-11-04T10:10:00Z")
        d = up.diff(a, b)
        self.assertEqual(d["claude"]["session_pct_delta"], 3)
        gw = [r for r in d["agy"] if r["group"] == "Gemini Models" and r["window"] == "weekly"][0]
        self.assertAlmostEqual(gw["remaining_pct_delta"], -10.0, places=3)

    def test_snapshot_survives_backend_failure(self):
        def run(argv, timeout=60):
            raise RuntimeError("boom")
        s = up.snapshot(["claude"], runner=run, now="2026-11-04T10:00:00Z")
        self.assertEqual(s["backends"]["claude"]["status"], "unavailable")

    def test_secret_scan_flags_credentials(self):
        up.assert_no_secrets({"a": "hello 42%"})
        for bad in ("sk-abcdefghijklmnopqrstuvwxyz1234", "Bearer abcdefghijklmnopqrstuvwxyz", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijk"):
            with self.assertRaises(ValueError):
                up.assert_no_secrets({"x": bad})

    def test_snapshot_passes_secret_scan(self):
        s = up.snapshot(["claude", "agy"], runner=self._runner(), now="2026-11-04T10:00:00Z")
        up.assert_no_secrets(s)


class TestTrackBudgetIntegration(unittest.TestCase):
    def _run(self, argv, root):
        import io
        from contextlib import redirect_stdout
        from unittest import mock
        import track_budget as tb
        buf = io.StringIO()
        with mock.patch.object(tb, "get_project_root", return_value=root), \
                mock.patch.object(sys, "argv", ["track_budget.py"] + argv), redirect_stdout(buf):
            tb.main()
        return buf.getvalue()

    def test_measured_record_lands_in_events_and_keeps_legacy_records(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(["record", "--task-id", "old", "--backend", "codex", "--tokens-est", "1000"], root)
            self._run(["record", "--task-id", "new", "--backend", "agy", "--tokens-est", "500", "--source", "measured",
                       "--task-class", "summarize", "--outcome", "ok", "--tokens-in", "400", "--tokens-out", "100",
                       "--concurrent"], root)
            ev = [json.loads(l) for l in (root / ".agent/learning/events.jsonl").read_text().splitlines()]
            self.assertNotIn("source", ev[0])
            self.assertEqual(ev[1]["source"], "measured")
            self.assertEqual(ev[1]["task_class"], "summarize")
            self.assertTrue(ev[1]["concurrent"])
            self.assertIn("Cumulative Tokens Consumed: 1,500", self._run(["status"], root))
            measured = up.load_measured(root)
            self.assertEqual(len(measured), 1)


class TestPredictor(unittest.TestCase):
    def _recs(self):
        out = []
        for i, t in enumerate([1000, 1100, 900, 1050, 950, 1000, 1020, 980, 1010, 990, 1030, 970]):
            out.append({"backend": "agy", "task_class": "summarize", "tokens_est": t, "timestamp": f"2026-11-04T10:{i:02d}:00Z"})
        return out

    def test_class_median_beats_static_ceiling(self):
        r = up.evaluate_predictor(self._recs(), min_train=8)
        self.assertEqual(r["status"], "ok")
        self.assertLess(r["mape_learned_pct"], 10)
        self.assertGreater(r["improvement_pct"], 20)

    def test_insufficient_data(self):
        self.assertEqual(up.evaluate_predictor(self._recs()[:5])["status"], "insufficient_data")

    def test_predict_fallback_order(self):
        m = up.fit_predictor(self._recs())
        self.assertEqual(up.predict(m, "agy", "summarize")["basis"], "class_median")
        self.assertEqual(up.predict(m, "agy", "edit")["basis"], "backend_median")
        self.assertEqual(up.predict(m, "claude", "edit")["basis"], "static_ceiling")

    def test_recommend_marks_constrained_backend_and_ranks_it_last(self):
        snap = {"backends": {"agy": {"buckets": [{"group": "G", "window": "weekly", "remaining_pct": 12.0, "resets_at": "x"}]},
                             "claude": {"session_pct_used": 2, "week_pct_used": 14},
                             "codex": {"five_hour_pct_used": 0, "weekly_pct_used": 0}}}
        recs = up.recommend(self._recs(), snap, "summarize", floor=15.0)
        by = {r["backend"]: r for r in recs}
        self.assertTrue(by["agy"]["constrained"])
        self.assertEqual(recs[-1]["backend"], "agy")
        self.assertEqual(by["agy"]["evidence_n"], 12)


class TestFuelGauge(unittest.TestCase):
    def _rec(self, tokens, pct, concurrent=False, backend="claude"):
        return {"backend": backend, "tokens_est": tokens, "concurrent": concurrent,
                "window_delta": {"claude": {"session_pct_delta": pct}}}

    def test_calibration_sums_across_tasks_and_skips_concurrent(self):
        recs = [self._rec(100000, 2), self._rec(150000, 3), self._rec(900000, 9, concurrent=True)]
        c = up.calibrate_capacity(recs, min_total_pct=5)
        self.assertEqual(c["status"], "ok")
        self.assertEqual(c["tokens_per_pct"], 50000)
        self.assertEqual(c["window_capacity_tokens"], 5000000)

    def test_calibration_refuses_when_percent_movement_too_small(self):
        c = up.calibrate_capacity([self._rec(100000, 2)], min_total_pct=5)
        self.assertEqual(c["status"], "insufficient_data")

    def test_runway_from_calibration(self):
        r = up.runway({"session_pct_used": 20}, {"status": "ok", "tokens_per_pct": 50000})
        self.assertEqual(r["est_tokens_left"], 4000000)
        self.assertIsNone(up.runway({"session_pct_used": 20}, {"status": "insufficient_data"}))

    def test_headroom_report_flags_low_window(self):
        snap = {"backends": {"agy": {"status": "ok", "buckets": [{"remaining_pct": 12.0, "group": "G", "window": "weekly", "resets_at": "x"}]},
                             "claude": {"status": "ok", "session_pct_used": 2, "week_pct_used": 14}}}
        rep = {r["backend"]: r for r in up.headroom_report(snap, 15.0)}
        self.assertTrue(rep["agy"]["constrained"])
        self.assertFalse(rep["claude"]["constrained"])

    def test_cost_known_and_unknown_prices(self):
        prices = {"m1": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}, "m2": {"input_per_mtok": None, "output_per_mtok": None}}
        self.assertEqual(up.cost_usd("m1", 1_000_000, 200_000, prices), 2.0)
        self.assertIsNone(up.cost_usd("m2", 1000, 1000, prices))
        self.assertIsNone(up.cost_usd("missing", 1000, 1000, prices))

    def test_shipped_price_table_never_guesses(self):
        prices = up.load_prices()
        self.assertIn("claude-haiku-4-5", prices)
        self.assertIsNone(prices["claude-sonnet-5-5"]["input_per_mtok"])


if __name__ == "__main__":
    unittest.main()
