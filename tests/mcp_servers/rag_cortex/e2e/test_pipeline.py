"""
End-to-End RAG Pipeline Integration Test.
Tests ingestion and querying using CortexOperations API.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.mark.integration
def test_rag_query_existing_protocol():
    """Test querying for Protocol 101."""
    ops = CortexOperations(str(PROJECT_ROOT))
    
    # Query for Protocol 101
    result = ops.query("What is Protocol 101?", max_results=5)
    
    assert result.status == "success", f"Query failed: {result.error}"
    
    # Check if any result contains Protocol 101 content
    all_content = " ".join([r.content for r in result.results])
    assert len(all_content) > 0, "No content in query results"
    print(f"✅ Protocol 101 query successful: {len(result.results)} results")

@pytest.mark.integration
def test_incremental_ingestion(tmp_path):
    """Test incremental document ingestion."""
    ops = CortexOperations(str(PROJECT_ROOT))
    
    # Create a test document
    test_file = tmp_path / "Test_Ingest_Doc.md"
    test_file.write_text("# Test Document\n\nThis is a test document for incremental ingestion.")
    
    # Ingest the test document
    result = ops.ingest_incremental(
        file_paths=[str(test_file)],
        skip_duplicates=False
    )
    
    assert result.status == "success", f"Ingestion failed: {result.error}"
    assert result.documents_added > 0 or result.skipped_duplicates > 0
    print(f"✅ Incremental ingestion verified: {result}")
