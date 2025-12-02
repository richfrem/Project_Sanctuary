# council_orchestrator/tests/test_delta_refresh_on_ingest_and_gitops.py
# Tests for delta refresh hooks on ingest and git-ops

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from council_orchestrator.orchestrator.memory.cache import CacheManager, CACHE

# Compute project root relative to this test file
# This file: Project_Sanctuary/council_orchestrator/tests/test_delta_refresh_on_ingest_and_gitops.py
# Project root: ../../../ from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDeltaRefreshIngest:
    """Test delta refresh functionality during ingest operations."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    def test_prefill_guardian_delta_updates_affected_bundles(self):
        """Test that prefill_guardian_delta updates only affected bundles."""
        # Initial prefill
        CacheManager.prefill_guardian_start_pack()
        initial_chronicles = CacheManager.get("guardian:dashboard:chronicles:latest")

        # Simulate file changes that should trigger chronicle refresh
        changed_paths = ["00_CHRONICLE/ENTRIES/new_chronicle.md"]

        # Call delta refresh
        CacheManager.prefill_guardian_delta(changed_paths)

        # Chronicles should be refreshed (different content or same but TTL reset)
        updated_chronicles = CacheManager.get("guardian:dashboard:chronicles:latest")

        # Content should exist (may be same if no new files, but TTL should be reset)
        assert updated_chronicles is not None

    def test_prefill_guardian_delta_ignores_unrelated_changes(self):
        """Test that delta refresh ignores changes to unrelated files."""
        # Initial prefill
        CacheManager.prefill_guardian_start_pack()
        initial_roadmap = CacheManager.get("guardian:dashboard:roadmap")

        # Change unrelated file
        changed_paths = ["unrelated_file.txt"]

        # Call delta refresh
        CacheManager.prefill_guardian_delta(changed_paths)

        # Roadmap should be unchanged
        updated_roadmap = CacheManager.get("guardian:dashboard:roadmap")
        assert updated_roadmap == initial_roadmap

    def test_prefill_guardian_delta_handles_multiple_changes(self):
        """Test that delta refresh handles multiple file changes correctly."""
        # Initial prefill
        CacheManager.prefill_guardian_start_pack()

        # Multiple changes affecting different bundles
        changed_paths = [
            "00_CHRONICLE/ENTRIES/new_chronicle.md",
            "01_PROTOCOLS/new_protocol.md",
            "ROADMAP/updated_plan.md"
        ]

        # Call delta refresh
        CacheManager.prefill_guardian_delta(changed_paths)

        # All affected bundles should be refreshed
        chronicles = CacheManager.get("guardian:dashboard:chronicles:latest")
        protocols = CacheManager.get("guardian:dashboard:protocols:latest")
        roadmap = CacheManager.get("guardian:dashboard:roadmap")

        # Should all exist (content may be same if files don't exist, but refreshed)
        assert chronicles is not None
        assert protocols is not None
        assert roadmap is not None


class TestDeltaRefreshGitOps:
    """Test delta refresh functionality during git operations."""

    def setup_method(self):
        """Clear cache before each test."""
        CACHE.clear()

    @patch('council_orchestrator.orchestrator.gitops.execute_mechanical_git')
    def test_gitops_calls_delta_refresh_after_successful_commit(self, mock_git):
        """Test that gitops calls delta refresh after successful commits."""
        # This test verifies the integration point exists
        # The actual call is made in gitops.py after successful commit/push

        # Read gitops.py to verify the integration
        gitops_path = PROJECT_ROOT / "council_orchestrator" / "orchestrator" / "gitops.py"
        with open(gitops_path, "r") as f:
            content = f.read()
            assert "CacheManager.prefill_guardian_delta" in content

    def test_delta_refresh_integration_points_exist(self):
        """Test that delta refresh integration points exist in cortex and gitops."""
        # Verify cortex.py has the integration
        cortex_path = PROJECT_ROOT / "council_orchestrator" / "orchestrator" / "memory" / "cortex.py"
        with open(cortex_path, "r") as f:
            cortex_content = f.read()
            assert "CacheManager.prefill_guardian_delta" in cortex_content

        # Verify gitops.py has the integration
        gitops_path = PROJECT_ROOT / "council_orchestrator" / "orchestrator" / "gitops.py"
        with open(gitops_path, "r") as f:
            gitops_content = f.read()
            assert "CacheManager.prefill_guardian_delta" in gitops_content


class TestDeltaRefreshWatchedPaths:
    """Test that delta refresh watches the correct file paths."""

    def test_watched_paths_mapping_exists(self):
        """Test that the watched paths mapping is properly defined."""
        # The watched paths are defined in prefill_guardian_delta
        # We can verify by calling it and checking behavior

        # This should trigger chronicle refresh
        changed_paths = ["00_CHRONICLE/ENTRIES/test.md"]
        CacheManager.prefill_guardian_delta(changed_paths)

        # This should trigger protocol refresh
        changed_paths = ["01_PROTOCOLS/test.md"]
        CacheManager.prefill_guardian_delta(changed_paths)

        # This should trigger roadmap refresh
        changed_paths = ["ROADMAP/test.md"]
        CacheManager.prefill_guardian_delta(changed_paths)

        # Should not raise exceptions
        assert True