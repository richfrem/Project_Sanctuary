import pytest
import threading
import time
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Mock components that try to connect to hardware/network
with patch('chromadb.HttpClient'), \
     patch('langchain_huggingface.HuggingFaceEmbeddings'), \
     patch('langchain_chroma.Chroma'), \
     patch('mcp_servers.rag_cortex.file_store.SimpleFileStore'):
    from mcp_servers.rag_cortex.operations import CortexOperations

class TestRaceConditions:
    """
    Protocol 128: Hardened Learning Loop - Race Condition Tests
    Verifies that the "Sandwich Validation" prevents snapshots when the
    repository state changes during the snapshot generation process.
    """
    
    @pytest.fixture
    def cortex_ops(self, tmp_path):
        # Setup a minimal mock environment
        # Create a mock client
        mock_client = MagicMock()
        
        with patch('chromadb.HttpClient', return_value=mock_client), \
             patch('langchain_huggingface.HuggingFaceEmbeddings'), \
             patch('langchain_chroma.Chroma'), \
             patch('mcp_servers.rag_cortex.file_store.SimpleFileStore'), \
             patch('mcp_servers.rag_cortex.operations.get_env_variable', return_value=None):
            ops = CortexOperations(
                project_root=str(tmp_path),
                client=mock_client
            )
        
        # Initialize Git repo in tmp_path
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@sanctuary.ai"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test Agent"], cwd=tmp_path, check=True)
        
        # Create a tracked file
        (tmp_path / "test_file.txt").write_text("Initial content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)
        
        return ops

    def test_race_condition_detection(self, cortex_ops, tmp_path):
        """
        Simulate a race condition where a file is modified during the snapshot 
        generation (between Pre-Flight and Post-Flight checks).
        """
        # 1. Create a dummy manifest
        manifest_files = ["test_file.txt"]
        
        # 2. Mock generate_snapshot to simulate delay + concurrent modification
        def mock_generate_with_interference(*args, **kwargs):
            logger.info("Mock snapshot started... sleeping to simulate work")
            # Create the file being mocked so the tool doesn't crash on next read
            output_file = kwargs.get('output_file')
            if output_file:
                 Path(output_file).write_text("# Initial Mock Content\n")
            
            time.sleep(1) # Simulate robust snapshot generation
            
            # SIMULATE ATTACK/RACE: Modify the repo state HERE
            logger.info(">>> INJECTING RACE CONDITION <<<")
            (tmp_path / "race_condition.txt").write_text("I AM A PHANTOM FILE")
            # We don't commit it, so it shows up as Untracked in git status
            
            return {"total_files": 1, "total_bytes": 100}

        # Setup mock for generate_snapshot
        with patch('mcp_servers.rag_cortex.operations.generate_snapshot', side_effect=mock_generate_with_interference):
            # 3. Execute Capture Snapshot
            response = cortex_ops.capture_snapshot(
                manifest_files=manifest_files,
                snapshot_type="audit",
                strategic_context="Testing Race Condition"
            )
            
        # 4. Assertions
        logger.info(f"Snapshot Response Status: {response.status}")
        logger.info(f"Snapshot Error Context: {response.git_diff_context}")
        
        # Expect FAIL due to integrity check
        assert response.status == "failed"
        assert "INTEGRITY FAILURE" in response.git_diff_context
        assert "Race condition detected" in response.git_diff_context
        assert "race_condition.txt" in response.git_diff_context

    def test_no_race_condition_success(self, cortex_ops, tmp_path):
        """
        Verify that snapshots succeed normally when no repository state 
        drift is detected during the operation.
        """
        # 1. Create a dummy manifest
        manifest_files = ["test_file.txt"]
        
        # 2. Mock generate_snapshot to simulate normal delay without modification
        def mock_generate_normal(*args, **kwargs):
            logger.info("Mock snapshot started...")
            time.sleep(0.5)
            # Create the file being mocked
            output_file = kwargs.get('output_file')
            if output_file:
                Path(output_file).write_text("# Mock Snapshot Content\n")
            return {"total_files": 1, "total_bytes": 100}

        with patch('mcp_servers.rag_cortex.operations.generate_snapshot', side_effect=mock_generate_normal):
            # 3. Execute Capture Snapshot
            response = cortex_ops.capture_snapshot(
                manifest_files=manifest_files,
                snapshot_type="audit",
                strategic_context="Testing Success Case"
            )
            
        # 4. Assertions
        logger.info(f"Snapshot Response Status: {response.status}")
        assert response.status == "success"
        assert response.manifest_verified is True
        assert "Verified" in response.git_diff_context
        assert "INTEGRITY FAILURE" not in response.git_diff_context

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
