"""
E2E Tests for sanctuary_cortex cluster (13 tools)

Tools tested:
- RAG Ingestion: cortex-ingest-full, cortex-ingest-incremental
- RAG Query: cortex-query, cortex-get-stats
- Cache (CAG): cortex-cache-stats, cortex-cache-get, cortex-cache-set, cortex-cache-warmup
- Protocol Tools: cortex-guardian-wakeup, cortex-learning-debrief, cortex-capture-snapshot
- Forge LLM: query-sanctuary-model, check-sanctuary-model-status

Note: cortex-ingest-full is slow (2-5 min). Tests use test fixtures only to protect .agent/learning/.
"""
import pytest
from pathlib import Path
from tests.mcp_servers.gateway.e2e.conftest import to_container_path


# Test fixtures directory - NEVER use .agent/learning/
TEST_FIXTURES_DIR = Path(__file__).parents[5] / "fixtures" / "test_docs"
CONTAINER_TEST_FIXTURES_DIR = to_container_path(TEST_FIXTURES_DIR)


# =============================================================================
# RAG INGESTION TOOLS (2)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCortexIngestionTools:
    
    def test_cortex_ingest_incremental(self, logged_call):
        """Test cortex-ingest-incremental with test fixtures."""
        sample_doc = CONTAINER_TEST_FIXTURES_DIR + "/sample_document.md"
        
        result = logged_call("sanctuary-cortex-cortex-ingest-incremental", {
            "file_paths": [sample_doc],
            "skip_duplicates": True
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    @pytest.mark.slow
    @pytest.mark.timeout(300)  # 5 minute timeout for full ingestion
    def test_cortex_ingest_full(self, logged_call):
        """
        Test cortex-ingest-full with test fixtures only.
        
        CRITICAL: Uses tests/fixtures/test_docs/ - NEVER .agent/learning/
        Per Task 148: This is slow (2-5 min), needs progress logging.
        """
        result = logged_call("sanctuary-cortex-cortex-ingest-full", {
            "source_directories": [CONTAINER_TEST_FIXTURES_DIR],
            "purge_existing": False  # Don't purge - just add test docs
        })
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# RAG QUERY TOOLS (2)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCortexQueryTools:
    
    def test_cortex_query(self, logged_call):
        """Test cortex-query performs semantic search."""
        result = logged_call("sanctuary-cortex-cortex-query", {
            "query": "What is the purpose of the test document?",
            "max_results": 3,
            "use_cache": False
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_cortex_get_stats(self, logged_call):
        """Test cortex-get-stats returns database statistics."""
        result = logged_call("sanctuary-cortex-cortex-get-stats", {})
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# CACHE (CAG) TOOLS (4)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCortexCacheTools:
    
    def test_cortex_cache_stats(self, logged_call):
        """Test cortex-cache-stats returns cache statistics."""
        result = logged_call("sanctuary-cortex-cortex-cache-stats", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_cortex_cache_set(self, logged_call):
        """Test cortex-cache-set stores a cached answer."""
        result = logged_call("sanctuary-cortex-cortex-cache-set", {
            "query": "E2E test query for cache",
            "answer": "E2E test answer stored at test time"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_cortex_cache_get(self, logged_call):
        """Test cortex-cache-get retrieves a cached answer."""
        result = logged_call("sanctuary-cortex-cortex-cache-get", {
            "query": "E2E test query for cache"
        })
        
        # May return cache miss if previous test didn't run - that's OK
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_cortex_cache_warmup(self, logged_call):
        """Test cortex-cache-warmup pre-populates cache."""
        result = logged_call("sanctuary-cortex-cortex-cache-warmup", {
            "genesis_queries": ["What is Sanctuary?", "How does the gateway work?"]
        })
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# PROTOCOL TOOLS (3)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCortexProtocolTools:
    
    def test_cortex_guardian_wakeup(self, logged_call):
        """Test cortex-guardian-wakeup generates boot digest."""
        result = logged_call("sanctuary-cortex-cortex-guardian-wakeup", {
            "mode": "minimal"  # Fast mode for testing
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_cortex_learning_debrief(self, logged_call):
        """Test cortex-learning-debrief scans for changes."""
        result = logged_call("sanctuary-cortex-cortex-learning-debrief", {
            "hours": 1  # Just last hour for speed
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_cortex_capture_snapshot(self, logged_call):
        """Test cortex-capture-snapshot generates snapshot."""
        result = logged_call("sanctuary-cortex-cortex-capture-snapshot", {
            "snapshot_type": "minimal",
            "strategic_context": "E2E test snapshot"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# FORGE LLM TOOLS (2)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCortexForgeLLMTools:
    
    def test_check_sanctuary_model_status(self, logged_call):
        """Test check-sanctuary-model-status returns model availability."""
        result = logged_call("sanctuary-cortex-check-sanctuary-model-status", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    @pytest.mark.slow
    @pytest.mark.timeout(90)
    def test_query_sanctuary_model(self, logged_call):
        """
        Test query-sanctuary-model queries Ollama model.
        
        Note: Depends on Ollama model availability.
        """
        result = logged_call("sanctuary-cortex-query-sanctuary-model", {
            "prompt": "Hello, this is an E2E test. Respond briefly.",
            "max_tokens": 50,
            "temperature": 0.1
        })
        
        # May fail if Ollama model not loaded - skip rather than fail
        if not result["success"] and "model" in str(result.get("error", "")).lower():
            pytest.skip("Ollama model not available")
        
        assert result["success"], f"Failed: {result.get('error')}"
