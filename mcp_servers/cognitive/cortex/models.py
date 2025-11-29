"""
Cortex MCP Server - Data Models

Pydantic models for RAG operations in the Mnemonic Cortex.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Ingest Full Models
# ============================================================================

@dataclass
class IngestFullRequest:
    """Request model for full ingestion."""
    purge_existing: bool = True
    source_directories: Optional[List[str]] = None


@dataclass
class IngestFullResponse:
    """Response model for full ingestion."""
    documents_processed: int
    chunks_created: int
    ingestion_time_ms: float
    vectorstore_path: str
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Query Models
# ============================================================================

@dataclass
class QueryRequest:
    """Request model for RAG query."""
    query: str
    max_results: int = 5
    use_cache: bool = False  # Phase 2 feature


@dataclass
class QueryResult:
    """Individual query result."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None


@dataclass
class QueryResponse:
    """Response model for RAG query."""
    results: List[QueryResult]
    query_time_ms: float
    status: str  # "success" or "error"
    cache_hit: bool = False  # Phase 2 feature
    error: Optional[str] = None


# ============================================================================
# Stats Models
# ============================================================================

@dataclass
class CollectionStats:
    """Statistics for a single collection."""
    count: int
    name: str


@dataclass
class StatsResponse:
    """Response model for database statistics."""
    total_documents: int
    total_chunks: int
    collections: Dict[str, CollectionStats]
    health_status: str  # "healthy", "degraded", or "error"
    cache_stats: Optional[Dict[str, Any]] = None  # Phase 2 feature
    error: Optional[str] = None


# ============================================================================
# Ingest Incremental Models
# ============================================================================

@dataclass
class IngestIncrementalRequest:
    """Request model for incremental ingestion."""
    file_paths: List[str]
    metadata: Optional[Dict[str, Any]] = None
    skip_duplicates: bool = True


@dataclass
class IngestIncrementalResponse:
    """Response model for incremental ingestion."""
    documents_added: int
    chunks_created: int
    skipped_duplicates: int
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Cache Operation Models (Protocol 114 - Guardian Wakeup)
# ============================================================================

@dataclass
class CacheGetResponse:
    """Response from cache retrieval operation."""
    cache_hit: bool
    answer: Optional[str]
    query_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class CacheSetResponse:
    """Response from cache storage operation."""
    cache_key: str
    stored: bool
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class CacheWarmupResponse:
    """Response from cache warmup operation."""
    queries_cached: int
    cache_hits: int
    cache_misses: int
    total_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class GuardianWakeupResponse:
    """Response from Guardian wakeup digest generation."""
    digest_path: str
    bundles_loaded: List[str]
    cache_hits: int
    cache_misses: int
    total_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if isinstance(value, list):
                result[field_name] = [to_dict(item) if hasattr(item, '__dataclass_fields__') else item for item in value]
            elif isinstance(value, dict):
                result[field_name] = {k: to_dict(v) if hasattr(v, '__dataclass_fields__') else v for k, v in value.items()}
            elif hasattr(value, '__dataclass_fields__'):
                result[field_name] = to_dict(value)
            else:
                result[field_name] = value
        return result
    return obj
