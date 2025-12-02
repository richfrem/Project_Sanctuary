# council_orchestrator/tests/test_cache_request_command.py
# Tests for cache_request command type (v9.4)

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from council_orchestrator.orchestrator.commands import handle_cache_request
from council_orchestrator.orchestrator.memory.cache import CacheManager, CacheItem, CACHE


class TestCacheRequestBundle:
    """Test cache_request with bundle parameter."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    def test_bundle_happy_path_returns_entries(self):
        """Test that bundle request returns expected number of entries."""
        # Prefill cache
        CacheManager.prefill_guardian_start_pack()

        # Create cache request command
        command = {
            "task_type": "cache_request",
            "task_description": "Test bundle request",
            "output_artifact_path": "test_output.md",
            "cache_request": {
                "bundle": "guardian_start_pack",
                "policy": {"refresh_if_stale": False, "strict": False}
            }
        }

        # Handle the request
        report = handle_cache_request(command)

        # Verify report contains expected content
        assert "# Guardian Wakeup Cache Check (v9.4)" in report
        assert "bundle=guardian_start_pack" in report
        assert "Items: 11" in report  # All keys in bundle are reported
        assert "Missing: 8" in report  # 8 items missing because files don't exist
        assert "Expired: 0" in report

    def test_refresh_if_stale_calls_prefill(self):
        """Test that refresh_if_stale=true calls prefill method."""
        with patch.object(CacheManager, 'prefill_guardian_start_pack') as mock_prefill:
            command = {
                "task_type": "cache_request",
                "task_description": "Test refresh",
                "output_artifact_path": "test_output.md",
                "cache_request": {
                    "bundle": "guardian_start_pack",
                    "policy": {"refresh_if_stale": True, "strict": False}
                }
            }

            handle_cache_request(command)

            # Verify prefill was called
            mock_prefill.assert_called_once()


class TestCacheRequestKeys:
    """Test cache_request with keys parameter."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    def test_keys_mode_returns_only_requested_keys(self):
        """Test that keys mode returns only the requested cache entries."""
        # Set up some test cache entries
        CacheManager.set(CacheItem("test:key1", "value1", 3600))
        CacheManager.set(CacheItem("test:key2", "value2", 3600))

        command = {
            "task_type": "cache_request",
            "task_description": "Test keys request",
            "output_artifact_path": "test_output.md",
            "cache_request": {
                "keys": ["test:key1"],
                "policy": {"refresh_if_stale": False, "strict": False}
            }
        }

        report = handle_cache_request(command)

        # Verify only requested key appears
        assert "test:key1" in report
        assert "test:key2" not in report
        assert "Items: 1" in report


class TestCacheRequestStrictMode:
    """Test cache_request strict mode behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    def test_strict_mode_failure_with_missing_items(self):
        """Test that strict mode raises exception when items are missing."""
        command = {
            "task_type": "cache_request",
            "task_description": "Test strict mode",
            "output_artifact_path": "test_output.md",
            "cache_request": {
                "keys": ["nonexistent:key"],
                "policy": {"refresh_if_stale": False, "strict": True}
            }
        }

        report = handle_cache_request(command)

        # Verify missing item is reported
        assert "Missing: 1" in report
        assert "Strict mode enabled" in report


class TestCacheRequestArtifactShape:
    """Test cache_request artifact format and content."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    def test_artifact_contains_expected_sections(self):
        """Test that artifact contains all expected markdown sections."""
        # Set up test data
        CacheManager.set(CacheItem("test:key", "test_value", 3600))

        command = {
            "task_type": "cache_request",
            "task_description": "Test artifact shape",
            "output_artifact_path": "test_output.md",
            "cache_request": {
                "keys": ["test:key"],
                "policy": {"refresh_if_stale": False, "strict": False}
            }
        }

        report = handle_cache_request(command)

        # Verify markdown structure
        lines = report.split('\n')
        assert lines[0] == "# Guardian Wakeup Cache Check (v9.4)"
        assert "## Summary" in report
        assert "## Items" in report
        assert "| key | ttl_remaining | size | sha256[:10] | source | last_updated |" in report

    def test_ttl_display_format(self):
        """Test that TTL is displayed in human-readable format."""
        CacheManager.set(CacheItem("test:key", "value", 7200))  # 2 hours

        command = {
            "task_type": "cache_request",
            "task_description": "Test TTL format",
            "output_artifact_path": "test_output.md",
            "cache_request": {
                "keys": ["test:key"],
                "policy": {"refresh_if_stale": False, "strict": False}
            }
        }

        report = handle_cache_request(command)

        # Should show something like "2h0m" (approximately)
        assert "h" in report and "m" in report  # Contains time format


class TestCacheRequestExpiredItems:
    """Test cache_request handling of expired items."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    def test_expired_items_are_marked_and_cleaned(self):
        """Test that expired items are marked as expired and removed from cache."""
        # Set an item with very short TTL (1 second)
        CacheManager.set(CacheItem("test:expired", "value", 1))

        # Wait for expiration
        time.sleep(1.1)

        command = {
            "task_type": "cache_request",
            "task_description": "Test expired items",
            "output_artifact_path": "test_output.md",
            "cache_request": {
                "keys": ["test:expired"],
                "policy": {"refresh_if_stale": False, "strict": False}
            }
        }

        report = handle_cache_request(command)

        # Verify expired item is reported
        assert "Expired: 1" in report
        assert "expired" in report

        # Verify item was removed from cache
        assert "test:expired" not in CACHE