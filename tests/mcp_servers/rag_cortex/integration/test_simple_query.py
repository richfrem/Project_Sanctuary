"""
Simple RAG integration test - tests the actual RAG pipeline.
Uses CortexOperations directly instead of subprocess calls.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.mark.integration
def test_rag_query_via_cortex_ops():
    """Test RAG query using CortexOperations."""
    ops = CortexOperations(str(PROJECT_ROOT))
    
    # Query for Protocol 101
    result = ops.query("What is Protocol 101?", max_results=3)
    
    assert result.status == "success", f"RAG query failed: {result.error}"
    assert len(result.results) > 0, "No results from RAG query"
    print(f"✅ RAG query successful: {len(result.results)} results returned")
