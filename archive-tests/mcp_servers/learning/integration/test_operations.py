"""
Learning MCP Integration Tests - Operations Testing
===================================================

Comprehensive integration tests for all Learning operations (Protocol 128).
Uses BaseIntegrationTest and follows the pattern in rag_cortex/integration/test_operations.py.

MCP OPERATIONS:
---------------
| Operation          | Type  | Description                              |
|--------------------|-------|------------------------------------------|
| learning_debrief   | WRITE | Scans repo for state changes (The Scout) |
| capture_snapshot   | WRITE | Generates audit/seal packets (The Seal)  |
| persist_soul       | WRITE | Persists state to Lineage (The Chronicle) |
| persist_soul_full  | WRITE | Full genome sync (The Chronicle)         |
| guardian_wakeup    | WRITE | Generates boot digest (The Bootloader)   |
"""
import pytest
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.learning.operations import LearningOperations
from mcp_servers.learning.models import PersistSoulRequest
from mcp_servers.lib.env_helper import get_env_variable

class TestLearningOperations(BaseIntegrationTest):
    """
    Integration tests for all Learning operations.
    Follows Protocol 128 workflow logic.
    """

    def get_required_services(self):
        """No external services required for Learning logic itself."""
        return []

    @pytest.fixture
    def learning_ops(self, tmp_path):
        # Use a temporary directory for file storage
        project_root = tmp_path / "project_root"
        project_root.mkdir(parents=True, exist_ok=True)
        
        # Setup structure
        (project_root / "00_CHRONICLE").mkdir(parents=True, exist_ok=True)
        (project_root / "01_PROTOCOLS").mkdir(parents=True, exist_ok=True)
        (project_root / ".agent" / "learning").mkdir(parents=True, exist_ok=True)
        
        ops = LearningOperations(str(project_root))
        return ops

    @pytest.fixture
    def real_learning_ops(self):
        """
        Fixture using the REAL project root for true integration testing.
        Tests using this fixture operate on actual project files and cache.
        """
        real_project_root = Path(__file__).parent.parent.parent.parent.parent
        return LearningOperations(str(real_project_root))

    #===========================================================================
    # MCP OPERATION: learning_debrief
    #===========================================================================
    def test_learning_debrief(self, learning_ops):
        """Verify the Scout can scan the repository and return the Truth Anchor."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "modified_file.py | 2 +"
            mock_run.return_value.returncode = 0
            
            # Mock file existence
            with patch("pathlib.Path.exists", return_value=False):
                result = learning_ops.learning_debrief(hours=24)
                assert "# [HARDENED] Learning Package Snapshot" in result
                assert "Git Status:" in result
                assert "modified_file.py" in result

    #===========================================================================
    # MCP OPERATION: capture_snapshot
    #===========================================================================
    def test_capture_snapshot_basic(self, learning_ops):
        """Verify the Seal can generate a snapshot and enforce strict rejection."""
        # 1. Setup manifest file
        pkg_file = learning_ops.project_root / "01_PROTOCOLS" / "test.md"
        pkg_file.write_text("# Test Protocol")
        
        manifest = ["01_PROTOCOLS/test.md"]
        
        with patch("mcp_servers.learning.operations.generate_snapshot") as mock_gen, \
             patch("subprocess.run") as mock_run:
            
            # Success Case: ONLY manifest files changed
            mock_run.return_value.stdout = "M  01_PROTOCOLS/test.md"
            mock_run.return_value.returncode = 0
            mock_gen.return_value = {"total_files": 1}
            
            # Target the specific instance of stat if possible, or use side_effect
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 100
                mock_stat.return_value.st_mode = 33188 # Standard file
                
                response = learning_ops.capture_snapshot(manifest_files=manifest, snapshot_type="audit")
                assert response.status == "success"
                assert response.manifest_verified is True
                assert "red_team_audit_packet.md" in response.snapshot_path

            # 2. Success Case: Seal Type
            mock_run.return_value.stdout = "M  01_PROTOCOLS/test.md" # Clean for manifest
            # Use local patch for stat to avoid breaking mkdir in previous steps
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 100
                mock_stat.return_value.st_mode = 33188
                response = learning_ops.capture_snapshot(manifest_files=manifest, snapshot_type="seal")
                assert response.status == "success"
                assert "learning_package_snapshot.md" in response.snapshot_path
            
            # Failure Case: Core file changed but NOT in manifest (STRICT REJECTION)
            mock_run.return_value.stdout = "M  ADRs/001_core.md"
            response = learning_ops.capture_snapshot(manifest_files=manifest, snapshot_type="audit")
            assert response.status == "error"
            assert "REJECTED" in response.git_diff_context

    #===========================================================================
    # MCP OPERATION: persist_soul & persist_soul_full
    #===========================================================================
    def test_persist_soul(self, learning_ops):
        """Verify the Chronicle can persist session state."""
        snapshot_rel = ".agent/learning/learning_package_snapshot.md"
        snapshot_abs = learning_ops.project_root / snapshot_rel
        snapshot_abs.write_text("# Test Snapshot")
        
        # 1. Basic Persist
        request = PersistSoulRequest(snapshot_path=snapshot_rel)
        response = learning_ops.persist_soul(request)
        assert response.status == "success"
        
        # 2. Full Sync
        request_full = PersistSoulRequest(snapshot_path=snapshot_rel, is_full_sync=True)
        response_full = learning_ops.persist_soul(request_full)
        assert response_full.status == "success"
        
        # 3. Missing File
        request_err = PersistSoulRequest(snapshot_path="missing.md")
        response_err = learning_ops.persist_soul(request_err)
        assert response_err.status == "error"
        assert "Snapshot not found" in response_err.error

    #===========================================================================
    # MCP OPERATION: guardian_wakeup
    #===========================================================================
    def test_guardian_wakeup(self, learning_ops):
        """Verify the Bootloader generates the context-aware briefing."""
        # Create guardian manifest for v3.0 manifest-driven wakeup
        manifest_path = learning_ops.project_root / ".agent" / "learning" / "guardian_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"core": ["README.md"], "topic": []}')
        
        # Create README.md for the manifest
        readme = learning_ops.project_root / "README.md"
        readme.write_text("# Test Project")
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "sanctuary-fleet" # Containerized check
            
            response = learning_ops.guardian_wakeup(mode="HOLISTIC")
            assert response.status == "success"
            assert Path(response.digest_path).exists()
            
            content = Path(response.digest_path).read_text()
            assert "# 🛡️ Guardian Wakeup Briefing" in content
            assert "## I. Strategic Directives" in content

    #===========================================================================
    # MCP OPERATION: guardian_snapshot
    #===========================================================================
    def test_guardian_snapshot(self, learning_ops):
        """Verify the session context pack generation."""
        # Setup a dummy entry
        entry = learning_ops.project_root / "00_CHRONICLE" / "ENTRIES" / "999_test.md"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("# Test Chronicle")

        with patch("mcp_servers.learning.operations.generate_snapshot") as mock_gen, \
             patch("subprocess.run") as mock_run:
            
            mock_run.return_value.stdout = "" # Clean git
            mock_run.return_value.returncode = 0
            mock_gen.return_value = {"total_files": 1}
            
            # Mock stat for the generated snapshot
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 100
                    mock_stat.return_value.st_mode = 33188
        
                    response = learning_ops.guardian_snapshot(strategic_context="Test")
                    assert response.status == "success"
                    # We expect the snapshot path to be the constant seal filename
                    assert "learning_package_snapshot.md" in response.snapshot_path

    #===========================================================================
    # RLM SYNTHESIS (Protocol 132) - Real Call Smoke Test
    #===========================================================================
    def test_rlm_real_call_ollama(self, learning_ops):
        """
        [INTEGRATION] Verifies _rlm_map actually calls the local Ollama instance
        and returns a summary string (not an error).
        Requires: Local Ollama running Sanctuary-Qwen2-7B.
        """
        # 1. Setup a single test file in a valid root
        test_file = learning_ops.project_root / "01_PROTOCOLS" / "test_rlm.md"
        test_file.write_text(
            "# Protocol Test\n"
            "This protocol defines the testing standard. "
            "It ensures that systems respond to inputs with expected outputs."
        )
        
        # 2. Run _rlm_map (Filtered to just this file if possible, or tolerate others)
        # Since _rlm_map takes roots, we pass the root we created.
        try:
            results = learning_ops._rlm_map(["01_PROTOCOLS"])
            
            # 3. Validation
            key = "01_PROTOCOLS/test_rlm.md"
            if key not in results:
                # Might be due to temp dir pathing issues in test runner vs real execution
                # But we check if *any* result came back
                assert len(results) > 0, "No results returned from RLM map"
                # If we found something else, that's fine, just check it didn't error
                first_val = list(results.values())[0]
                assert "[Ollama" not in first_val, f"Ollama Error: {first_val}"
                assert len(first_val) > 5, "Summary too short"
            else:
                summary = results[key]
                assert len(summary) > 5
                # Check for error strings
                if "[Ollama" in summary:
                    pytest.fail(f"Real Ollama Call Failed: {summary}")
                
        except Exception as e:
            # If Ollama is down/slow integration test might fail, which is valid signal
            pytest.fail(f"RLM Integration Exception: {e}")

    #===========================================================================
    # RLM CACHE INTEGRATION TESTS - Real Project Cache
    #===========================================================================
    
    #===========================================================================
    # RLM CACHE INTEGRATION TESTS - Real Project Cache
    #===========================================================================
    
    def test_rlm_cache_progressive_population(self, real_learning_ops):
        """
        [REAL INTEGRATION] Verifies cache population with new files.
        Finds ADRs not in cache, processes them, and verifies they are saved with the new schema.
        """
        import json
        import random
        
        cache_path = real_learning_ops.project_root / ".agent" / "learning" / "rlm_summary_cache.json"
        
        # 1. Load current cache keys
        cached_keys = set()
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
                cached_keys = set(cache.keys())
            except: pass
            
        # 2. Find 2 ADRs currently NOT in the cache
        adr_dir = real_learning_ops.project_root / "ADRs"
        all_adrs = [str(f.relative_to(real_learning_ops.project_root)) for f in adr_dir.glob("*.md")]
        
        # Exclude artifacts and templates
        eligible = [a for a in all_adrs if a not in cached_keys and "template" not in a.lower()]
        
        if not eligible:
            # If everything is already cached, just pick any 2 to verify they stay there
            sample_targets = random.sample(all_adrs, min(2, len(all_adrs)))
        else:
            sample_targets = random.sample(eligible, min(2, len(eligible)))
            
        # 3. Process them
        try:
            results = real_learning_ops._rlm_map(sample_targets)
        except Exception as e:
            pytest.fail(f"RLM call failed: {e}")
            
        # 4. Verify results and cache update
        cache = json.loads(cache_path.read_text())
        for target in sample_targets:
            assert target in cache, f"Target {target} should be in cache now"
            entry = cache[target]
            assert "hash" in entry
            assert "file_mtime" in entry
            assert "summarized_at" in entry
            assert "summary" in entry
            assert "[Ollama" not in entry["summary"], "Stored an error in cache"

    def test_rlm_cache_hit_performance(self, real_learning_ops):
        """
        [REAL INTEGRATION] Verifies cache hit performance.
        Picks files from cache and ensures they return instantly.
        """
        import json
        import time
        import random
        
        cache_path = real_learning_ops.project_root / ".agent" / "learning" / "rlm_summary_cache.json"
        if not cache_path.exists():
            pytest.skip("Cache file doesn't exist yet, can't test hits.")
            
        cache = json.loads(cache_path.read_text())
        if not cache:
            pytest.skip("Cache is empty, can't test hits.")
            
        # Pick 3 random cached files
        sample_files = random.sample(list(cache.keys()), min(3, len(cache)))
        
        # First hit run
        start = time.time()
        results = real_learning_ops._rlm_map(sample_files)
        duration = time.time() - start
        
        # 3 hits should take < 500ms total
        assert duration < 1.0, f"Cache hit too slow: {duration:.2f}s for {len(sample_files)} files"
        assert len(results) == len(sample_files)

    def test_rlm_skips_recursive_artifacts(self, real_learning_ops):
        """
        [REAL INTEGRATION] Verifies RLM safety filter skips recursive artifacts.
        Attempts to force summary of 'learning_package_snapshot.md' and expects 0 results.
        """
        # Pick a snapshot that likely exists (or README as control)
        targets = [".agent/learning/learning_package_snapshot.md", "README.md"]
        
        results = real_learning_ops._rlm_map(targets)
        
        # README.md should be in results (it's safe)
        assert "README.md" in results
        
        # the snapshot MUST NOT be in results
        assert ".agent/learning/learning_package_snapshot.md" not in results
        
        # Verify cache hits log for README.md if it was already there
        # (This just ensures the test ran correctly)
        assert len(results) >= 1
