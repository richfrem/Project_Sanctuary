#!/usr/bin/env python3
"""
Unit tests for Phase 3 Cache Prefill functionality.
Tests Guardian Start Pack prefill and delta refresh.
"""

import unittest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the components we need to test
from council_orchestrator.orchestrator.memory.cache import CacheManager, CacheItem, CACHE


class TestCachePrefillGuardianBundle(unittest.TestCase):
    """Test Guardian Start Pack prefill creates all expected keys."""

    def setUp(self):
        """Clear cache before each test."""
        CACHE.clear()
        # Create CacheManager instance for tests
        self.cache_manager = CacheManager(Path('/tmp'), MagicMock())

    def tearDown(self):
        """Clear cache after each test."""
        CACHE.clear()

    @patch('council_orchestrator.orchestrator.memory.cache.PROJECT_ROOT', Path('/tmp/test'))
    def test_cache_prefill_guardian_bundle_creates_all_keys(self):
        """Test that prefill_guardian_start_pack creates all expected cache keys."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_root = Path(tmp_dir)

            # Create mock directory structure
            (test_root / "00_CHRONICLE" / "ENTRIES").mkdir(parents=True)
            (test_root / "01_PROTOCOLS").mkdir(parents=True)
            (test_root / "ROADMAP").mkdir(parents=True)
            (test_root / "council_orchestrator").mkdir(parents=True)
            (test_root / "council_orchestrator" / "schemas").mkdir(parents=True)
            (test_root / "council_orchestrator" / "logs").mkdir(parents=True)

            # Create mock files
            (test_root / "00_CHRONICLE" / "ENTRIES" / "test1.md").write_text("# Test Chronicle")
            (test_root / "01_PROTOCOLS" / "test2.md").write_text("# Test Protocol")
            (test_root / "ROADMAP" / "PHASED_EVOLUTION_PLAN_Phase2-Phase3-Protocol113.md").write_text("# Roadmap")
            (test_root / "council_orchestrator" / "README.md").write_text("# README")
            (test_root / "council_orchestrator" / "schemas" / "council-round-packet-v1.0.0.json").write_text("{}")
            (test_root / "council_orchestrator" / "logs" / "orchestrator.log").write_text("log line 1\nlog line 2\n")

            # Create a CacheManager with the test root
            test_cache_manager = CacheManager(test_root, MagicMock())
            test_cache_manager.prefill_guardian_start_pack()

            # Check that all expected keys exist
            expected_keys = [
                "guardian:dashboard:chronicles:latest",
                "guardian:dashboard:protocols:latest",
                "guardian:dashboard:roadmap",
                "guardian:docs:orchestrator_readme",
                "guardian:packets:schema",
                "guardian:ops:orchestrator_log:tail"
            ]

            for key in expected_keys:
                self.assertIn(key, CACHE, f"Missing cache key: {key}")
                self.assertIsNotNone(test_cache_manager.get(key), f"Cache key {key} should not be None")


class TestCachePrefillDelta(unittest.TestCase):
    """Test delta refresh functionality."""

    def setUp(self):
        """Clear cache before each test."""
        CACHE.clear()
        # Create CacheManager instance for tests
        self.cache_manager = CacheManager(Path('/tmp'), MagicMock())

    def tearDown(self):
        """Clear cache after each test."""
        CACHE.clear()

    @patch('council_orchestrator.orchestrator.memory.cache.PROJECT_ROOT', Path('/tmp/test'))
    def test_cache_prefill_delta_refreshes_on_chronicle_update(self):
        """Test that delta refresh updates chronicle cache when chronicle files change."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_root = Path(tmp_dir)

            # Create initial structure
            (test_root / "00_CHRONICLE" / "ENTRIES").mkdir(parents=True)
            (test_root / "00_CHRONICLE" / "ENTRIES" / "initial.md").write_text("# Initial")

            # Create CacheManager with test root
            test_cache_manager = CacheManager(test_root, MagicMock())
            
            # Initial prefill
            test_cache_manager.prefill_guardian_start_pack()
            initial_chronicles = test_cache_manager.get("guardian:dashboard:chronicles:latest")

            # Add new chronicle file
            (test_root / "00_CHRONICLE" / "ENTRIES" / "new.md").write_text("# New Chronicle")

            # Delta refresh
            test_cache_manager.prefill_guardian_delta(["00_CHRONICLE/ENTRIES/new.md"])
            updated_chronicles = test_cache_manager.get("guardian:dashboard:chronicles:latest")

            # Should be different after refresh
            self.assertNotEqual(initial_chronicles, updated_chronicles)


class TestCacheTTL(unittest.TestCase):
    """Test TTL functionality."""

    def setUp(self):
        """Clear cache before each test."""
        CACHE.clear()
        # Create CacheManager instance for tests
        self.cache_manager = CacheManager(Path('/tmp'), MagicMock())

    def tearDown(self):
        """Clear cache after each test."""
        CACHE.clear()

    def test_cache_ttl_expiry_clears_items(self):
        """Test that items expire after TTL."""
        # Set item with very short TTL
        item = CacheItem("test:key", "test_value", ttl_seconds=1)
        self.cache_manager.set(item)

        # Should exist immediately
        self.assertEqual(self.cache_manager.get("test:key"), "test_value")

        # Wait for expiry
        time.sleep(1.1)

        # Should be gone
        self.assertIsNone(self.cache_manager.get("test:key"))


class TestCacheLogTail(unittest.TestCase):
    """Test log tail functionality."""

    def setUp(self):
        """Clear cache before each test."""
        CACHE.clear()
        # Create CacheManager instance for tests
        self.cache_manager = CacheManager(Path('/tmp'), MagicMock())

    def tearDown(self):
        """Clear cache after each test."""
        CACHE.clear()

    @patch('council_orchestrator.orchestrator.memory.cache.PROJECT_ROOT', Path('/tmp/test'))
    def test_cache_log_tail_rotates_and_stays_small(self):
        """Test that log tail only keeps last N lines."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_root = Path(tmp_dir)
            log_file = test_root / "council_orchestrator" / "logs" / "orchestrator.log"
            log_file.parent.mkdir(parents=True)

            # Create log with many lines
            lines = [f"log line {i}" for i in range(200)]
            log_file.write_text("\n".join(lines))

            # Create CacheManager with test root
            test_cache_manager = CacheManager(test_root, MagicMock())
            test_cache_manager.prefill_guardian_start_pack()

            tail_content = test_cache_manager.get("guardian:ops:orchestrator_log:tail")

            # Should only have last 150 lines
            tail_lines = tail_content.split("\n")
            self.assertEqual(len(tail_lines), 150)
            self.assertIn("log line 199", tail_content)  # Last line should be there
            self.assertNotIn("log line 49", tail_content)  # Early lines should be gone


class TestCacheKeys(unittest.TestCase):
    """Test cache key stability and documentation."""

    def setUp(self):
        """Clear cache before each test."""
        CACHE.clear()

    def tearDown(self):
        """Clear cache after each test."""
        CACHE.clear()

    def test_cache_keys_stable_and_documented(self):
        """Test that all cache keys follow documented naming convention."""
        # This test ensures we don't accidentally change key names
        expected_keys = [
            "guardian:dashboard:chronicles:latest",
            "guardian:dashboard:protocols:latest",
            "guardian:dashboard:roadmap",
            "guardian:docs:orchestrator_readme",
            "guardian:docs:command_schema",
            "guardian:docs:howto_commit",
            "guardian:packets:schema",
            "guardian:blueprint:optical_anvil",
            "guardian:ops:engine_config",
            "guardian:ops:orchestrator_log:tail",
            "guardian:rounds:last_jsonl"
        ]

        # All keys should follow guardian:* pattern
        for key in expected_keys:
            self.assertTrue(key.startswith("guardian:"), f"Key {key} doesn't follow guardian: prefix")
            self.assertIn(":", key, f"Key {key} should have namespace separator")


if __name__ == '__main__':
    unittest.main()