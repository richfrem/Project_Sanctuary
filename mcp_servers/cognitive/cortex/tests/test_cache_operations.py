"""
Tests for Cortex cache operations (Protocol 114 - Guardian Wakeup).
"""
import pytest
from pathlib import Path
from mcp_servers.cognitive.cortex.operations import CortexOperations


@pytest.fixture
def cortex_ops():
    """Fixture providing CortexOperations instance."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    return CortexOperations(str(project_root))


def test_cache_get_miss(cortex_ops):
    """Test cache get with no cached value (cache miss)."""
    response = cortex_ops.cache_get("nonexistent query xyz123 unique")
    
    assert response.cache_hit == False
    assert response.answer is None
    assert response.status == "success"
    assert response.query_time_ms >= 0


def test_cache_set_and_get(cortex_ops):
    """Test cache set followed by get (cache hit)."""
    test_query = "test query for cache operations"
    test_answer = "test answer from cache"
    
    # Set
    set_response = cortex_ops.cache_set(test_query, test_answer)
    assert set_response.stored == True
    assert set_response.status == "success"
    assert len(set_response.cache_key) > 0
    
    # Get
    get_response = cortex_ops.cache_get(test_query)
    assert get_response.cache_hit == True
    assert get_response.answer == test_answer
    assert get_response.status == "success"


def test_cache_warmup_default_queries(cortex_ops):
    """Test cache warmup with default genesis queries."""
    response = cortex_ops.cache_warmup()
    
    assert response.status == "success"
    assert response.queries_cached == 10  # Default genesis queries
    assert response.cache_hits + response.cache_misses == response.queries_cached
    assert response.total_time_ms > 0


def test_cache_warmup_custom_queries(cortex_ops):
    """Test cache warmup with custom query list."""
    custom_queries = [
        "What is Protocol 87?",
        "What is Protocol 101?"
    ]
    
    response = cortex_ops.cache_warmup(genesis_queries=custom_queries)
    
    assert response.status == "success"
    assert response.queries_cached == 2
    assert response.cache_hits + response.cache_misses == 2


def test_guardian_wakeup(cortex_ops):
    """Test Guardian wakeup digest generation."""
    response = cortex_ops.guardian_wakeup()
    
    assert response.status == "success"
    assert len(response.bundles_loaded) == 3
    assert "chronicles" in response.bundles_loaded
    assert "protocols" in response.bundles_loaded
    assert "roadmap" in response.bundles_loaded
    assert response.cache_hits + response.cache_misses == 3
    assert response.total_time_ms > 0
    
    # Verify digest file was created
    digest_path = Path(response.digest_path)
    assert digest_path.exists()
    assert digest_path.name == "guardian_boot_digest.md"
    
    # Verify digest content
    content = digest_path.read_text()
    assert "# Guardian Boot Digest" in content
    assert "## Chronicles" in content
    assert "## Protocols" in content
    assert "## Roadmap" in content


def test_cache_operations_error_handling(cortex_ops):
    """Test error handling in cache operations."""
    # Test with invalid inputs
    response = cortex_ops.cache_get("")
    assert response.status == "success"  # Empty query is valid, just returns miss
    
    response = cortex_ops.cache_set("", "")
    assert response.status == "success"  # Empty values are valid
