"""Tests for the checkout/checkin usage hook (MISSION-035). Uses injected snapshots; never calls real CLIs."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import usage_hook  # noqa: E402


def _snap(ts, claude_pct=2, agy_remaining=50.0):
    return {"timestamp": ts, "backends": {
        "claude": {"status": "ok", "session_pct_used": claude_pct, "week_pct_used": 14},
        "agy": {"status": "ok", "buckets": [{"group": "Gemini Models", "bucket": "w", "window": "weekly",
                                            "remaining_pct": agy_remaining, "resets_at": "x"}]},
        "codex": {"status": "ok", "five_hour_pct_used": 0.0, "weekly_pct_used": 0.0}}}


class TestHook(unittest.TestCase):
    def test_disabled_by_env_returns_none(self):
        self.assertEqual(os.environ.get("SANCTUARY_NO_USAGE"), "1")  # conftest isolation
        self.assertIsNone(usage_hook.start_snapshot())

    def test_start_snapshot_uses_injected_fn_when_enabled(self):
        old = os.environ.pop("SANCTUARY_NO_USAGE")
        try:
            s = usage_hook.start_snapshot(snapshot_fn=lambda b: _snap("t0"))
            self.assertEqual(s["timestamp"], "t0")
        finally:
            os.environ["SANCTUARY_NO_USAGE"] = old

    def test_probe_failure_never_raises(self):
        old = os.environ.pop("SANCTUARY_NO_USAGE")
        try:
            def boom(b):
                raise RuntimeError("down")
            self.assertIsNone(usage_hook.start_snapshot(snapshot_fn=boom))
        finally:
            os.environ["SANCTUARY_NO_USAGE"] = old

    def test_headroom_warning_flags_low_own_backend(self):
        snap = _snap("t", 2, 12.0)
        w = usage_hook.headroom_warnings(snap, "antigravity", floor=15.0)
        self.assertEqual([x["backend"] for x in w], ["agy"])
        self.assertTrue(w[0]["own_backend"])
        self.assertEqual(usage_hook.headroom_warnings(None, "antigravity"), [])
        self.assertFalse(usage_hook.headroom_warnings(_snap("t", 2, 60.0), "claude-code"))

    def test_backend_for_agent(self):
        self.assertEqual(usage_hook.backend_for_agent("claude-code"), "claude")
        self.assertEqual(usage_hook.backend_for_agent("antigravity"), "agy")
        self.assertEqual(usage_hook.backend_for_agent("codex-astra"), "codex")
        self.assertIsNone(usage_hook.backend_for_agent("someone"))

    def test_finish_with_vendor_tokens_computes_deltas_and_wall(self):
        old = os.environ.pop("SANCTUARY_NO_USAGE")
        try:
            lease = {"usage_start": _snap("2026-11-04T10:00:00Z", 2, 50.0), "checked_out_timestamp": time.time() - 120}
            with tempfile.TemporaryDirectory() as d:
                out = usage_hook.finish(Path(d), lease, "MISSION-X", "antigravity", task_class="refactor", tokens=12345,
                                        quality=2, snapshot_fn=lambda b: _snap("2026-11-04T10:02:00Z", 5, 40.0), record=False)
            self.assertEqual(out["tokens"], 12345)
            self.assertEqual(out["tokens_source"], "vendor_reported")
            self.assertGreaterEqual(out["wall_seconds"], 120)
            self.assertEqual(out["window_delta"]["claude"]["session_pct_delta"], 3)
            self.assertEqual(out["quality"], 2)
        finally:
            os.environ["SANCTUARY_NO_USAGE"] = old

    def test_finish_cost_known_and_unknown_model(self):
        old = os.environ.pop("SANCTUARY_NO_USAGE")
        try:
            lease = {"usage_start": _snap("2026-11-04T10:00:00Z"), "checked_out_timestamp": time.time() - 5}
            with tempfile.TemporaryDirectory() as d:
                known = usage_hook.finish(Path(d), lease, "M", "antigravity", tokens=1_000_000, model="claude-haiku-4-5",
                                          snapshot_fn=lambda b: _snap("2026-11-04T10:01:00Z"), record=False)
                unknown = usage_hook.finish(Path(d), lease, "M", "antigravity", tokens=1_000_000, model="claude-sonnet-5-5",
                                            snapshot_fn=lambda b: _snap("2026-11-04T10:01:00Z"), record=False)
            self.assertEqual(known["cost_usd"], 1.0)  # 1M input-side tokens at $1/MTok; output unknown for vendor totals
            self.assertIsNone(unknown["cost_usd"])
        finally:
            os.environ["SANCTUARY_NO_USAGE"] = old

    def test_finish_noop_when_disabled_or_no_lease(self):
        self.assertIsNone(usage_hook.finish(Path("."), {"checked_out_timestamp": 1}, "M", "a"))
        self.assertIsNone(usage_hook.finish(Path("."), {}, "M", "a", no_usage=True))


if __name__ == "__main__":
    unittest.main()


class TestFuelGaugeInBoot(unittest.TestCase):
    def test_relay_shows_gauge_and_low_flag_from_cache(self):
        import relay
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            usage_hook.save_latest(root, _snap("2026-11-04T10:00:00Z", 2, 12.0))
            md = relay.render_markdown(relay.build_relay(root))
            self.assertIn("Fuel gauge", md)
            self.assertIn("agy 12% LOW", md)
            self.assertIn("claude 86%", md)

    def test_relay_prompts_for_reading_when_cache_missing_or_stale(self):
        import relay
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIn("no reading under 6 h old", relay.render_markdown(relay.build_relay(root)))
            usage_hook.save_latest(root, _snap("t"))
            self.assertIsNone(usage_hook.load_latest(root, now=time.time() + 7 * 3600))
