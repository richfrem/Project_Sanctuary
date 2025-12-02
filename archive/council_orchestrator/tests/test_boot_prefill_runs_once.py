# council_orchestrator/tests/test_boot_prefill_runs_once.py
# Tests for boot prefill idempotency

import pytest
from pathlib import Path
from unittest.mock import patch
from council_orchestrator.orchestrator.memory.cache import CacheManager, CACHE

# Compute project root relative to this test file
# This file: Project_Sanctuary/council_orchestrator/tests/test_boot_prefill_runs_once.py
# Project root: ../../../ from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestBootPrefillIdempotency:
    """Test that boot prefill runs once and is idempotent."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()
        self.cache_manager = CacheManager()

    def test_prefill_guardian_start_pack_populates_cache(self):
        """Test that prefill_guardian_start_pack populates the cache with expected keys."""
        self.cache_manager.prefill_guardian_start_pack()

        # Should have populated some cache entries
        assert len(CACHE) > 0

        # Should contain expected keys (at least the ones that have files)
        keys = list(CACHE.keys())
        assert any("guardian:dashboard:chronicles:latest" in key for key in keys) or \
               "guardian:dashboard:chronicles:latest" in keys
        assert any("guardian:dashboard:protocols:latest" in key for key in keys) or \
               "guardian:dashboard:protocols:latest" in keys
        assert any("guardian:dashboard:roadmap" in key for key in keys) or \
               "guardian:dashboard:roadmap" in keys

    def test_prefill_guardian_start_pack_is_idempotent(self):
        """Test that running prefill multiple times doesn't create duplicates."""
        # First run
        self.cache_manager.prefill_guardian_start_pack()
        first_run_keys = set(CACHE.keys())
        first_run_count = len(CACHE)

        # Second run
        self.cache_manager.prefill_guardian_start_pack()
        second_run_keys = set(CACHE.keys())
        second_run_count = len(CACHE)

        # Should be the same (idempotent)
        assert first_run_keys == second_run_keys
        assert first_run_count == second_run_count

    def test_prefill_guardian_start_pack_sets_ttl(self):
        """Test that prefill sets appropriate TTL values."""
        self.cache_manager.prefill_guardian_start_pack()

        # Check that entries have expiration times set
        for key, entry in CACHE.items():
            assert "expires_at" in entry
            assert entry["expires_at"] > 0
            # Should expire in future (TTL set)
            import time
            assert entry["expires_at"] > time.time()

    @patch('council_orchestrator.orchestrator.memory.cache.PROJECT_ROOT', new=Path("/fake/path"))
    def test_prefill_handles_missing_files_gracefully(self):
        """Test that prefill handles missing files without crashing."""
        # With fake project root, files won't exist but prefill should not crash
        try:
            self.cache_manager.prefill_guardian_start_pack()
            # Should not raise exception
            assert True
        except Exception as e:
            pytest.fail(f"Prefill should handle missing files gracefully, but got: {e}")

    def test_prefill_runs_on_orchestrator_boot(self):
        """Test that prefill is called during orchestrator boot."""
        # This is tested by the fact that main.py calls cache_manager.prefill_guardian_start_pack()
        # We can verify this by checking that the call exists in main.py
        main_path = PROJECT_ROOT / "council_orchestrator" / "orchestrator" / "main.py"
        with open(main_path, "r") as f:
            content = f.read()
            assert "cache_manager.prefill_guardian_start_pack()" in content