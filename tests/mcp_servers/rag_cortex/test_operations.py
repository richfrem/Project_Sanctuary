"""
Unit tests for Cortex MCP operations

Note: These are integration-style tests that require the actual
Mnemonic Cortex infrastructure to be set up. They are marked
with pytest.mark.integration and can be skipped in CI.
"""
import pytest
import tempfile
import os
from pathlib import Path
from mcp_servers.rag_cortex.operations import CortexOperations


def test_operations_init(temp_project_root):
    """Test operations initialization."""
    ops = CortexOperations(temp_project_root)
    assert ops.project_root == Path(temp_project_root)
    assert ops.scripts_dir == Path(temp_project_root) / "mcp_servers" / "rag_cortex" / "scripts"


@pytest.mark.integration
@pytest.mark.skip(reason="Skipped per user request (full ingest)")
def test_ingest_full_script_not_found(temp_project_root):
    """Test ingest_full when script doesn't exist."""
    ops = CortexOperations(temp_project_root)
    response = ops.ingest_full()
    
    assert response.status == "error"
    assert "not found" in response.error.lower()


@pytest.mark.integration
def test_query_error_handling(temp_project_root):
    """Test query error handling when service not available."""
    ops = CortexOperations(temp_project_root)
    response = ops.query("test query")
    
    # Should return error response when database doesn't exist
    assert response.status == "error"
    assert response.error is not None


@pytest.mark.integration
def test_get_stats_no_database(temp_project_root):
    """Test get_stats when database doesn't exist."""
    ops = CortexOperations(temp_project_root)
    response = ops.get_stats()
    
    # Should return error or degraded status
    assert response.health_status in ["error", "degraded"]


@pytest.mark.integration
def test_ingest_incremental_error_handling(temp_project_root):
    """Test ingest_incremental error handling."""
    ops = CortexOperations(temp_project_root)
    
    # Try to ingest non-existent file
    response = ops.ingest_incremental(
        file_paths=["nonexistent.md"],
        skip_duplicates=True
    )
    
    # Should return success with error message (no valid files)
    assert response.status == "success"
    assert response.error == "No valid files to ingest"
    assert response.documents_added == 0


# The following tests would require actual Mnemonic Cortex setup
# and are marked as integration tests

@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists("/Users/richardfremmerlid/Projects/Project_Sanctuary/mnemonic_cortex"),
    reason="Requires actual Mnemonic Cortex setup"
)
def test_get_stats_real_database():
    """Test get_stats with real database (integration test)."""
    project_root = "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    ops = CortexOperations(project_root)
    response = ops.get_stats()
    
    # Should return healthy status if database exists
    if response.health_status == "healthy":
        assert response.total_documents > 0
        assert response.total_chunks > 0
        assert "child_chunks" in response.collections
        assert "parent_documents" in response.collections


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists("/Users/richardfremmerlid/Projects/Project_Sanctuary/mnemonic_cortex"),
    reason="Requires actual Mnemonic Cortex setup"
)
def test_query_real_database():
    """Test query with real database (integration test)."""
    project_root = "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    ops = CortexOperations(project_root)
    response = ops.query("What is Protocol 101?", max_results=3)
    
    # Should return successful response
    if response.status == "success":
        assert len(response.results) > 0
        assert response.query_time_ms > 0
        assert all(hasattr(r, 'content') for r in response.results)
        assert all(hasattr(r, 'metadata') for r in response.results)
