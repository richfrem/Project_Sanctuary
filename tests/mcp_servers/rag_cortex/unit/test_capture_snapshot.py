
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.fixture
def mock_cortex_ops():
    with patch("mcp_servers.rag_cortex.operations.CortexOperations.__init__", return_value=None):
        ops = CortexOperations("dummy_root")
        ops.project_root = Path("/dummy/root")
        ops.data_dir = Path("/dummy/root/data")
        return ops

def test_capture_snapshot_audit_success(mock_cortex_ops):
    """Verify audit snapshot creation with valid inputs."""
    manifest = ["file1.py", "file2.md"]
    context = "Security Audit"
    
    with patch("subprocess.run") as mock_run, \
         patch("mcp_servers.rag_cortex.operations.generate_snapshot") as mock_gen_snapshot, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("pathlib.Path.mkdir"), \
         patch("json.dump"):
         
        # Mock git diff return (all files verified)
        # Note: subprocess.run is still used for git diff
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "file1.py\nfile2.md"
        
        response = mock_cortex_ops.capture_snapshot(
            manifest_files=manifest,
            snapshot_type="audit",
            strategic_context=context
        )
        
        assert response.status == "success"
        assert response.snapshot_type == "audit"
        assert response.manifest_verified is True
        assert "Verified: 2 files" in response.git_diff_context

        # Verify generate_snapshot call
        mock_gen_snapshot.assert_called_once()
        args, kwargs = mock_gen_snapshot.call_args
        assert kwargs['project_root'] == Path("/dummy/root")
        assert "manifest_audit" in str(kwargs['manifest_path'])
        assert kwargs['output_file'].name == "red_team_audit_packet.md"

def test_capture_snapshot_seal_default_manifest(mock_cortex_ops):
    """Verify seal snapshot uses learning_manifest.json by default."""
    manifest_content = ["manifest_file.md"]
    
    with patch("subprocess.run") as mock_run, \
         patch("mcp_servers.rag_cortex.operations.generate_snapshot") as mock_gen_snapshot, \
         patch("builtins.open", mock_open(read_data='["manifest_file.md"]')), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.unlink"), \
         patch("pathlib.Path.mkdir"), \
         patch("json.dump") as mock_json_dump, \
         patch("json.load", return_value=manifest_content):
         
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        
        response = mock_cortex_ops.capture_snapshot(
            manifest_files=[], # Empty manifest
            snapshot_type="seal"
        )
        
        assert response.status == "success"
        # Verify that json.dump was called with the content from the mock learning_manifest.json
        # The tool writes a temp manifest file.
        args, _ = mock_json_dump.call_args
        assert args[0] == manifest_content # Verify it loaded the default

def test_capture_snapshot_git_mismatch(mock_cortex_ops):
    """Verify behavior when manifest files are not in git diff (audit mode)."""
    manifest = ["new_file.py"] # Not in git diff
    
    with patch("subprocess.run") as mock_run, \
         patch("mcp_servers.rag_cortex.operations.generate_snapshot") as mock_gen_snapshot, \
         patch("builtins.open", mock_open()), \
         patch("pathlib.Path.mkdir"), \
         patch("json.dump"):
         
        # Mock git diff returning empty
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "" 
        
        response = mock_cortex_ops.capture_snapshot(
            manifest_files=manifest,
            snapshot_type="audit"
        )
        
        assert response.status == "success" # Still succeeds
        assert response.manifest_verified is True # True because no critical omissions
        assert "Shadow Manifest" in response.git_diff_context
