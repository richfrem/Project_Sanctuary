"""
Test enhanced get_stats with sample retrieval (from inspect_db.py).
"""
import pytest
from mcp_servers.cognitive.cortex.operations import CortexOperations
from mcp_servers.cognitive.cortex.models import DocumentSample


@pytest.mark.integration
def test_get_stats_with_samples():
    """Test get_stats with sample document retrieval."""
    from pathlib import Path
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    ops = CortexOperations(str(project_root))
    
    # Get stats with samples
    result = ops.get_stats(include_samples=True, sample_count=3)
    
    # Verify basic stats
    assert result.status == "success" or result.health_status in ["healthy", "degraded"]
    assert result.total_documents >= 0
    assert result.total_chunks >= 0
    
    # Verify samples if database has data
    if result.total_chunks > 0 and result.samples:
        assert isinstance(result.samples, list)
        assert len(result.samples) <= 3  # Should respect sample_count
        
        # Verify sample structure
        for sample in result.samples:
            assert isinstance(sample, DocumentSample)
            assert hasattr(sample, 'id')
            assert hasattr(sample, 'metadata')
            assert hasattr(sample, 'content_preview')
            assert isinstance(sample.metadata, dict)
            assert isinstance(sample.content_preview, str)
            # Content preview should be truncated to ~150 chars
            assert len(sample.content_preview) <= 154  # 150 + "..."
    
    print(f"✅ get_stats with samples: {len(result.samples) if result.samples else 0} samples retrieved")


@pytest.mark.integration  
def test_get_stats_without_samples():
    """Test get_stats without sample retrieval (default behavior)."""
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    ops = CortexOperations(str(project_root))
    
    # Get stats without samples (default)
    result = ops.get_stats(include_samples=False)
    
    # Verify basic stats
    assert result.status == "success" or result.health_status in ["healthy", "degraded", "error"]
    assert result.total_documents >= 0
    assert result.total_chunks >= 0
    
    # Verify no samples returned
    assert result.samples is None
    
    print(f"✅ get_stats without samples: {result.total_documents} docs, {result.total_chunks} chunks")


if __name__ == "__main__":
    print("Running enhanced get_stats tests...")
    test_get_stats_with_samples()
    test_get_stats_without_samples()
    print("✅ All enhanced get_stats tests passed!")
