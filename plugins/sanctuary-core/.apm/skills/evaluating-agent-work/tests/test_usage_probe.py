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


def _tx_cache(path, ts, cw_total, cw_1h, cread):
    with open(path, "w") as f:
        f.write(json.dumps({"type": "assistant", "timestamp": ts, "message": {"role": "assistant", "usage": {
            "input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": cw_total, "cache_read_input_tokens": cread,
            "cache_creation": {"ephemeral_1h_input_tokens": cw_1h, "ephemeral_5m_input_tokens": cw_total - cw_1h}}}}) + "\n")


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


def _tc2(ts, total, inp=None, cached=0, out=0):
    return {"timestamp": ts, "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {
        "input_tokens": inp if inp is not None else total, "cached_input_tokens": cached, "output_tokens": out, "total_tokens": total}}}}


class TestCodexTokensBetween(unittest.TestCase):
    def test_delta_of_cumulative_totals_within_interval(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "2026/11/04").mkdir(parents=True)
            _rollout(d / "2026/11/04/rollout-a.jsonl", [_tc2("2026-11-04T10:00:00Z", 1000), _tc2("2026-11-04T10:10:00Z", 5000),
                                                         _tc2("2026-11-04T10:20:00Z", 9000)])
            r = up.codex_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z")
        self.assertEqual(r["total_tokens"], 4000)  # 5000 at end minus 1000 at start
        self.assertFalse(r["concurrent"])

    def test_session_started_inside_interval_counts_from_zero_and_two_sessions_flag_concurrent(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "2026/11/04").mkdir(parents=True)
            _rollout(d / "2026/11/04/rollout-a.jsonl", [_tc2("2026-11-04T10:06:00Z", 700), _tc2("2026-11-04T10:09:00Z", 1200)])
            _rollout(d / "2026/11/04/rollout-b.jsonl", [_tc2("2026-11-04T10:07:00Z", 50)])
            r = up.codex_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z")
        self.assertEqual(r["total_tokens"], 1250)
        self.assertTrue(r["concurrent"])

    def test_input_cached_and_output_are_split_across_the_interval(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "2026/11/04").mkdir(parents=True)
            _rollout(d / "2026/11/04/rollout-a.jsonl", [_tc2("2026-11-04T10:00:00Z", 1000, inp=900, cached=100, out=100),
                                                         _tc2("2026-11-04T10:10:00Z", 6000, inp=5400, cached=2100, out=600)])
            r = up.codex_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z")
        self.assertEqual(r["total_tokens"], 5000)
        self.assertEqual((r["input_tokens"], r["cached_input_tokens"], r["output_tokens"]), (4500, 2000, 500))

    def test_no_activity_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            r = up.codex_tokens_between(Path(d), "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z")
        self.assertEqual(r["total_tokens"], 0)
        self.assertFalse(r["concurrent"])


class TestCacheSplit(unittest.TestCase):
    def test_transcript_splits_cache_write_by_ttl(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _tx_cache(d / "s1.jsonl", "2026-11-04T10:10:00Z", cw_total=1000, cw_1h=700, cread=5000)
            r = up.claude_tokens_between(d, "2026-11-04T10:05:00Z", "2026-11-04T10:15:00Z")
        self.assertEqual(r["cache_creation_input_tokens"], 1000)
        self.assertEqual(r["cache_write_1h_tokens"], 700)
        self.assertEqual(r["cache_read_input_tokens"], 5000)

    def test_cost_prices_1h_cache_write_at_its_own_rate(self):
        prices = {"m": {"input_per_mtok": 2.0, "output_per_mtok": 10.0, "cached_input_per_mtok": 0.1,
                        "cache_write_per_mtok": 2.5, "cache_write_1h_per_mtok": 4.0}}
        # 1M 5m-write*2.5 + 1M 1h-write*4.0
        self.assertAlmostEqual(up.cost_usd("m", 0, 0, prices, cache_write=1_000_000, cache_write_1h=1_000_000), 6.5)
        prices["m"]["cache_write_1h_per_mtok"] = None
        self.assertIsNone(up.cost_usd("m", 0, 0, prices, cache_write_1h=1000))

    @unittest.skipUnless(up.find_catalog_dir() and "claude-sonnet-5-5" in up.load_prices(), "catalog with claude-sonnet-5-5 not installed")
    def test_real_session_cost_matches_vendor_reported_total(self):
        # Real Claude Code session on claude-sonnet-5-5 reported $29.55 (4.0k in, 592.9k out, 175.5M cache read, 1.5M cache write at the 1h TTL).
        c = up.cost_usd("claude-sonnet-5-5", 4000, 592_900, cache_read=175_500_000, cache_write_1h=1_500_000)
        self.assertLess(abs(c - 29.55) / 29.55, 0.01)


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

    def _catalog(self, d):
        d = Path(d)
        for name, models in (("claude-models.json", [{"cli_id": "m-known", "status": "GA", "pricing_usd_per_1m": {
                                "input": 2.0, "output": 10.0, "cached_input": 0.2, "cache_write": 2.5}},
                               {"cli_id": "m-noprice", "status": "available", "pricing_usd_per_1m": {"input": None, "output": None}}]),
                             ("codex-models.json", []), ("agy-models.json", []),
                             ("copilot-models.json", [{"cli_id": "m-known", "status": "GA", "pricing_usd_per_1m": {
                                "input": 9.0, "output": 9.0}, "credits_per_1m": {"input": 900}}])):
            (d / name).write_text(json.dumps({"_meta": {"updated": "2026-09-13"}, "models": models}))
        return d

    def test_prices_come_from_cli_agent_catalog_not_a_local_table(self):
        with tempfile.TemporaryDirectory() as d:
            prices = up.load_prices(self._catalog(d))
        self.assertEqual(prices["m-known"]["input_per_mtok"], 2.0)
        self.assertEqual(prices["copilot:m-known"]["credits_per_mtok"], {"input": 900})  # copilot never shadows direct API
        self.assertNotIn("m-missing", prices)

    def test_cost_exact_id_with_cache_rates_and_never_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            prices = up.load_prices(self._catalog(d))
        # 1M in*$2 + 0.5M out*$10 + 2M cache-read*$0.2 + 1M cache-write*$2.5 = 2 + 5 + 0.4 + 2.5
        self.assertAlmostEqual(up.cost_usd("m-known", 1_000_000, 500_000, prices, cache_read=2_000_000, cache_write=1_000_000), 9.9)
        self.assertIsNone(up.cost_usd("m-noprice", 1000, 1000, prices))
        self.assertIsNone(up.cost_usd("m-known-5", 1000, 1000, prices))  # similar name is not a match

    def test_no_local_price_table_shipped(self):
        self.assertFalse((HERE.parent / "references" / "model-prices.json").exists())

    @unittest.skipUnless(up.find_catalog_dir(), "cli-agent catalog not installed")
    def test_real_catalog_loads(self):
        prices = up.load_prices()
        self.assertGreater(len(prices), 10)
        self.assertEqual(prices["claude-haiku-4-5"]["output_per_mtok"], 5.0)


if __name__ == "__main__":
    unittest.main()


class TestAgyJsonUsage(unittest.TestCase):
    RAW = ('{"conversation_id":"c","status":"SUCCESS","response":"x","duration_seconds":14.2,"num_turns":1,'
           '"usage":{"input_tokens":5510,"output_tokens":502,"thinking_tokens":0,"cache_read_tokens":33091,"total_tokens":6012}}')

    def test_parses_split_from_agy_json_output(self):
        u = up.parse_agy_json_usage(self.RAW)
        self.assertEqual((u["total_tokens"], u["input_tokens"], u["output_tokens"], u["cache_read_tokens"]), (6012, 5510, 502, 33091))

    def test_returns_none_for_text_failure_or_missing_usage(self):
        self.assertIsNone(up.parse_agy_json_usage("plain text answer"))
        self.assertIsNone(up.parse_agy_json_usage('{"status":"ERROR"}'))
        self.assertIsNone(up.parse_agy_json_usage(""))


class TestOpenRouterLedger(unittest.TestCase):
    """OpenRouter has no free usage probe; the zero-token source is the cli-agent spend ledger."""

    def _ledger(self, d, **fields):
        p = Path(d) / "ledger.json"
        p.write_text(json.dumps(fields), encoding="utf-8")
        return str(p)

    def test_reads_spend_and_cap_from_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            r = up.read_openrouter_ledger(self._ledger(d, cumulative_spend_usd=2.5, hard_cap_usd=10.0, last_updated="t"))
        self.assertEqual((r["status"], r["spend_usd"], r["cap_usd"], r["remaining_pct"]), ("ok", 2.5, 10.0, 75.0))

    def test_explicit_cap_and_env_cap_precedence(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            p = self._ledger(d, cumulative_spend_usd=1.0, hard_cap_usd=10.0)
            self.assertEqual(up.read_openrouter_ledger(p, cap_usd=4.0)["remaining_pct"], 75.0)
            old = os.environ.get("OPENROUTER_MONTHLY_CAP_USD")
            os.environ["OPENROUTER_MONTHLY_CAP_USD"] = "20"
            try:
                self.assertEqual(up.read_openrouter_ledger(p)["cap_usd"], 20.0)
            finally:
                if old is None:
                    del os.environ["OPENROUTER_MONTHLY_CAP_USD"]
                else:
                    os.environ["OPENROUTER_MONTHLY_CAP_USD"] = old

    def test_overspend_clamps_to_zero_and_bad_cap_uses_default(self):
        with tempfile.TemporaryDirectory() as d:
            r = up.read_openrouter_ledger(self._ledger(d, cumulative_spend_usd=12.0, hard_cap_usd=-1))
        self.assertEqual((r["cap_usd"], r["remaining_pct"]), (10.0, 0.0))

    def test_missing_or_corrupt_ledger_is_unavailable_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(up.read_openrouter_ledger(str(Path(d) / "nope.json"))["status"], "unavailable")
            bad = Path(d) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(up.read_openrouter_ledger(str(bad))["status"], "unavailable")
            self.assertEqual(up.read_openrouter_ledger(self._ledger(d, other=1))["status"], "unavailable")

    def test_remaining_pct_and_headroom_flag_constrained(self):
        data = {"status": "ok", "remaining_pct": 8.0}
        self.assertEqual(up._remaining_pct(data, "openrouter"), 8.0)
        self.assertIsNone(up._remaining_pct({"status": "unavailable"}, "openrouter"))
        rep = up.headroom_report({"backends": {"openrouter": data}}, floor=15.0)
        self.assertTrue(rep[0]["constrained"])

    def test_snapshot_wrapper_adds_openrouter_and_keeps_requested_order(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("OPENROUTER_SPEND_LEDGER")
            os.environ["OPENROUTER_SPEND_LEDGER"] = self._ledger(d, cumulative_spend_usd=1.0, hard_cap_usd=10.0)
            try:
                snap = up.snapshot(["openrouter"])
                self.assertEqual(list(snap["backends"]), ["openrouter"])
                self.assertEqual(snap["backends"]["openrouter"]["remaining_pct"], 90.0)
                snap2 = up.snapshot(["local", "openrouter"])
                self.assertEqual(list(snap2["backends"]), ["local", "openrouter"])
            finally:
                if old is None:
                    del os.environ["OPENROUTER_SPEND_LEDGER"]
                else:
                    os.environ["OPENROUTER_SPEND_LEDGER"] = old
