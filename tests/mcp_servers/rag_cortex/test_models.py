"""
Unit tests for Cortex MCP models
"""
import pytest
from mcp_servers.rag_cortex.models import (
    IngestFullRequest,
    IngestFullResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    StatsResponse,
    CollectionStats,
    IngestIncrementalRequest,
    IngestIncrementalResponse,
    to_dict
)


def test_ingest_full_request():
    """Test IngestFullRequest model."""
    request = IngestFullRequest(
        purge_existing=True,
        source_directories=["01_PROTOCOLS", "00_CHRONICLE"]
    )
    assert request.purge_existing is True
    assert request.source_directories == ["01_PROTOCOLS", "00_CHRONICLE"]


def test_ingest_full_response():
    """Test IngestFullResponse model."""
    response = IngestFullResponse(
        documents_processed=459,
        chunks_created=2145,
        ingestion_time_ms=45230.5,
        vectorstore_path="/path/to/chroma_db",
        status="success"
    )
    assert response.documents_processed == 459
    assert response.chunks_created == 2145
    assert response.status == "success"


def test_query_request():
    """Test QueryRequest model."""
    request = QueryRequest(
        query="What is Protocol 101?",
        max_results=5,
        use_cache=False
    )
    assert request.query == "What is Protocol 101?"
    assert request.max_results == 5
    assert request.use_cache is False


def test_query_result():
    """Test QueryResult model."""
    result = QueryResult(
        content="Full document content",
        metadata={"source_file": "01_PROTOCOLS/101.md"},
        relevance_score=0.95
    )
    assert result.content == "Full document content"
    assert result.metadata["source_file"] == "01_PROTOCOLS/101.md"
    assert result.relevance_score == 0.95


def test_query_response():
    """Test QueryResponse model."""
    results = [
        QueryResult(
            content="Content 1",
            metadata={"source_file": "file1.md"}
        )
    ]
    response = QueryResponse(
        results=results,
        query_time_ms=234.5,
        cache_hit=False,
        status="success"
    )
    assert len(response.results) == 1
    assert response.query_time_ms == 234.5
    assert response.status == "success"


def test_collection_stats():
    """Test CollectionStats model."""
    stats = CollectionStats(count=2145, name="child_chunks_v5")
    assert stats.count == 2145
    assert stats.name == "child_chunks_v5"


def test_stats_response():
    """Test StatsResponse model."""
    collections = {
        "child_chunks": CollectionStats(count=2145, name="child_chunks_v5"),
        "parent_documents": CollectionStats(count=459, name="parent_documents_v5")
    }
    response = StatsResponse(
        total_documents=459,
        total_chunks=2145,
        collections=collections,
        health_status="healthy"
    )
    assert response.total_documents == 459
    assert response.total_chunks == 2145
    assert response.health_status == "healthy"


def test_ingest_incremental_request():
    """Test IngestIncrementalRequest model."""
    request = IngestIncrementalRequest(
        file_paths=["file1.md", "file2.md"],
        metadata={"author": "test"},
        skip_duplicates=True
    )
    assert len(request.file_paths) == 2
    assert request.metadata["author"] == "test"
    assert request.skip_duplicates is True


def test_ingest_incremental_response():
    """Test IngestIncrementalResponse model."""
    response = IngestIncrementalResponse(
        documents_added=3,
        chunks_created=15,
        skipped_duplicates=1,
        status="success"
    )
    assert response.documents_added == 3
    assert response.chunks_created == 15
    assert response.skipped_duplicates == 1
    assert response.status == "success"


def test_to_dict():
    """Test to_dict helper function."""
    response = IngestFullResponse(
        documents_processed=10,
        chunks_created=50,
        ingestion_time_ms=1000.0,
        vectorstore_path="/path",
        status="success"
    )
    result = to_dict(response)
    assert isinstance(result, dict)
    assert result["documents_processed"] == 10
    assert result["chunks_created"] == 50
    assert result["status"] == "success"


def test_to_dict_with_nested_objects():
    """Test to_dict with nested dataclass objects."""
    collections = {
        "child_chunks": CollectionStats(count=100, name="child_chunks_v5")
    }
    response = StatsResponse(
        total_documents=10,
        total_chunks=100,
        collections=collections,
        health_status="healthy"
    )
    result = to_dict(response)
    assert isinstance(result, dict)
    assert isinstance(result["collections"], dict)
    assert result["collections"]["child_chunks"]["count"] == 100
