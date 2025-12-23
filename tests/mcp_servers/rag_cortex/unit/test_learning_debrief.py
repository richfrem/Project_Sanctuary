
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from datetime import datetime
from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.fixture
def mock_cortex_ops():
    with patch("mcp_servers.rag_cortex.operations.CortexOperations.__init__", return_value=None):
        ops = CortexOperations("dummy_root")
        ops.project_root = Path("/dummy/root")
        ops.data_dir = Path("/dummy/root/data")
        return ops

def test_learning_debrief_with_snapshot(mock_cortex_ops):
    """Verify learning_debrief picks up a recent snapshot."""
    snapshot_content = "# Test Snapshot Content"
    mtime = datetime.now().timestamp() - 1800 # 30 mins ago
    
    def exists_side_effect(self_path):
        # Only return True for the snapshot and primer for testing
        return "learning_package_snapshot.md" in str(self_path) or "cognitive_primer.md" in str(self_path)

    with patch("pathlib.Path.exists", side_effect=exists_side_effect, autospec=True), \
         patch("pathlib.Path.stat") as mock_stat, \
         patch("pathlib.Path.read_text", return_value=snapshot_content), \
         patch("subprocess.run") as mock_run:
         
        mock_stat.return_value.st_mtime = mtime
        mock_run.return_value.stdout = "modified_file.py | 2 +"
        
        # We need to mock _get_recency_delta as well
        with patch.object(CortexOperations, "_get_recency_delta", return_value="Recency Delta Info"):
            result = mock_cortex_ops.learning_debrief(hours=24)
            
            assert "# [DRAFT] Learning Package Snapshot v3.5" in result
            assert "✅ Loaded Learning Package Snapshot from 0.5h ago." in result
            assert snapshot_content in result

def test_learning_debrief_no_snapshot(mock_cortex_ops):
    """Verify learning_debrief handles missing snapshot gracefully."""
    with patch("pathlib.Path.exists", return_value=False), \
         patch("subprocess.run") as mock_run:
         
        mock_run.return_value.stdout = "No uncommitted changes."
        
        with patch.object(CortexOperations, "_get_recency_delta", return_value="No recent files"):
            result = mock_cortex_ops.learning_debrief(hours=24)
            
            assert "ℹ️ No `.agent/learning/learning_package_snapshot.md` detected." in result
            assert "⚠️ No active Learning Package Snapshot found." in result
