"""
End-to-End RAG Pipeline Integration Test.
Tests ingestion and querying using robust patterns (direct imports + subprocess).
"""
import pytest
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@pytest.mark.integration
def test_rag_query_existing_protocol():
    """
    Test querying for an existing protocol (Protocol 101).
    This verifies the retrieval pipeline works on pre-existing data.
    """
    import subprocess
    
    # Use subprocess to run the query command (simulating CLI/MCP usage)
    result = subprocess.run(
        [sys.executable, "mnemonic_cortex/app/main.py", "What is Protocol 101?"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, f"Query failed: {result.stderr}"
    
    # Check for key phrases from Protocol 101
    output = result.stdout
    assert "Unbreakable Commit" in output or "Doctrine" in output, \
        f"Query output did not contain expected Protocol 101 terms. Got:\n{output}"
    
    print(f"✅ Protocol 101 query successful")

@pytest.mark.integration
def test_incremental_ingestion(tmp_path):
    """
    Test incremental ingestion of a new document.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from mcp_servers.rag_cortex.operations import CortexOperations
    
    # 1. Create a dummy file
    test_file = tmp_path / "Test_Ingest_Doc.md"
    test_file.write_text("# Test Document\n\nThis is a test document for incremental ingestion.")
    
    # 2. Ingest it
    ops = CortexOperations(str(PROJECT_ROOT))
    result = ops.ingest_incremental(
        file_paths=[str(test_file)],
        skip_duplicates=False
    )
    
    # 3. Verify ingestion success
    assert result.status == "success"
    assert result.documents_added > 0 or result.skipped_duplicates > 0
    
    print(f"✅ Incremental ingestion verified: {result}")
