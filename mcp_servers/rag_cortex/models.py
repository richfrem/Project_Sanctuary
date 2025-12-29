#============================================
# mcp_servers/rag_cortex/models.py
# Purpose: Pydantic/Dataclass models for RAG operations in the Mnemonic Cortex.
# Role: Single Source of Truth
# Used as a module by operations.py and server.py
# Calling example:
#   from mcp_servers.rag_cortex.models import to_dict
# LIST OF MODELS:
#   - IngestFullRequest
#   - IngestFullResponse
#   - QueryRequest
#   - QueryResult
#   - QueryResponse
#   - DocumentSample
#   - CollectionStats
#   - StatsResponse
#   - IngestIncrementalRequest
#   - IngestIncrementalResponse
#   - CacheGetResponse
#   - CacheSetResponse
#   - CacheWarmupResponse
#   - GuardianWakeupResponse
#   - CaptureSnapshotRequest
#   - CaptureSnapshotResponse
# LIST OF FUNCTIONS:
#   - to_dict
#============================================

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Ingest Full Models
# ============================================================================

@dataclass
class IngestFullRequest:
    #============================================
    # Model: IngestFullRequest
    # Purpose: Request model for full ingestion.
    # Fields:
    #   purge_existing: Whether to purge existing database
    #   source_directories: Optional specific directories to ingest
    #============================================
    purge_existing: bool = True
    source_directories: Optional[List[str]] = None


@dataclass
class IngestFullResponse:
    #============================================
    # Model: IngestFullResponse
    # Purpose: Response model for full ingestion.
    #============================================
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
    #============================================
    # Model: QueryRequest
    # Purpose: Request model for RAG query.
    #============================================
    query: str
    max_results: int = 5
    use_cache: bool = False  # Phase 2 feature


@dataclass
class QueryResult:
    #============================================
    # Model: QueryResult
    # Purpose: Individual query result.
    #============================================
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None


@dataclass
class QueryResponse:
    #============================================
    # Model: QueryResponse
    # Purpose: Response model for RAG query.
    #============================================
    results: List[QueryResult]
    query_time_ms: float
    status: str  # "success" or "error"
    cache_hit: bool = False  # Phase 2 feature
    error: Optional[str] = None


# ============================================================================
# Stats Models
# ============================================================================

@dataclass
class DocumentSample:
    #============================================
    # Model: DocumentSample
    # Purpose: Sample document for diagnostics.
    #============================================
    id: str
    metadata: Dict[str, Any]
    content_preview: str  # First 150 chars


@dataclass
class CollectionStats:
    #============================================
    # Model: CollectionStats
    # Purpose: Statistics for a single collection.
    #============================================
    count: int
    name: str


@dataclass
class StatsResponse:
    #============================================
    # Model: StatsResponse
    # Purpose: Response model for database statistics.
    #============================================
    total_documents: int
    total_chunks: int
    collections: Dict[str, CollectionStats]
    health_status: str  # "healthy", "degraded", or "error"
    samples: Optional[List[DocumentSample]] = None  # Enhanced diagnostics from inspect_db
    cache_stats: Optional[Dict[str, Any]] = None  # Phase 2 feature
    error: Optional[str] = None


# ============================================================================
# Ingest Incremental Models
# ============================================================================

@dataclass
class IngestIncrementalRequest:
    #============================================
    # Model: IngestIncrementalRequest
    # Purpose: Request model for incremental ingestion.
    #============================================
    file_paths: List[str]
    metadata: Optional[Dict[str, Any]] = None
    skip_duplicates: bool = True


@dataclass
class IngestIncrementalResponse:
    #============================================
    # Model: IngestIncrementalResponse
    # Purpose: Response model for incremental ingestion.
    #============================================
    documents_added: int
    chunks_created: int
    skipped_duplicates: int
    ingestion_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Cache Operation Models (Protocol 114 - Guardian Wakeup)
# ============================================================================

@dataclass
class CacheGetResponse:
    #============================================
    # Model: CacheGetResponse
    # Purpose: Response from cache retrieval operation.
    #============================================
    cache_hit: bool
    answer: Optional[str]
    query_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class CacheSetResponse:
    #============================================
    # Model: CacheSetResponse
    # Purpose: Response from cache storage operation.
    #============================================
    cache_key: str
    stored: bool
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class CacheWarmupResponse:
    #============================================
    # Model: CacheWarmupResponse
    # Purpose: Response from cache warmup operation.
    #============================================
    queries_cached: int
    cache_hits: int
    cache_misses: int
    total_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class GuardianWakeupResponse:
    #============================================
    # Model: GuardianWakeupResponse
    # Purpose: Response from Guardian wakeup digest generation.
    #============================================
    digest_path: str
    bundles_loaded: List[str]
    cache_hits: int
    cache_misses: int
    total_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Capture Snapshot Models (Protocol 128 v3.5)
# ============================================================================

@dataclass
class CaptureSnapshotRequest:
    #============================================
    # Model: CaptureSnapshotRequest
    # Purpose: Request model for tool-driven snapshotting.
    #============================================
    manifest_files: List[str]
    snapshot_type: str = "audit"  # "audit" or "seal"
    strategic_context: Optional[str] = None


@dataclass
class CaptureSnapshotResponse:
    #============================================
    # Model: CaptureSnapshotResponse
    # Purpose: Response model for tool-driven snapshotting.
    #============================================
    snapshot_path: str
    manifest_verified: bool
    git_diff_context: str
    snapshot_type: str
    status: str  # "success" or "error"
    total_files: int = 0
    total_bytes: int = 0
    error: Optional[str] = None


# ============================================================================
# Persist Soul Models (ADR 079 - Johnny Appleseed)
# ============================================================================

@dataclass
class PersistSoulRequest:
    #============================================
    # Model: PersistSoulRequest
    # Purpose: Request to broadcast the soul to the AI Commons.
    #============================================
    snapshot_path: str = ".agent/learning/learning_package_snapshot.md"
    valence: float = 0.0
    uncertainty: float = 0.0
    is_full_sync: bool = False


@dataclass
class PersistSoulResponse:
    #============================================
    # Model: PersistSoulResponse
    # Purpose: Response from the persistence operation.
    #============================================
    status: str  # "success", "quarantined", or "error"
    repo_url: str
    snapshot_name: str
    error: Optional[str] = None


# FastMCP Request Models
# ============================================================================
from pydantic import BaseModel, Field

class CortexIngestFullRequest(BaseModel):
    purge_existing: bool = Field(True, description="Whether to purge existing data")
    source_directories: Optional[List[str]] = Field(None, description="Paths to directories to ingest")

class CortexQueryRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    max_results: int = Field(5, description="Maximum number of context fragments")
    use_cache: bool = Field(False, description="Whether to use Mnemonic Cache")
    reasoning_mode: bool = Field(False, description="Whether to use reasoning model")

class CortexIngestIncrementalRequest(BaseModel):
    file_paths: List[str] = Field(..., description="Paths to files to ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for documents")
    skip_duplicates: bool = Field(True, description="Skip if already in store")

class CortexCacheGetRequest(BaseModel):
    query: str = Field(..., description="Query key to look up")

class CortexCacheSetRequest(BaseModel):
    query: str = Field(..., description="Query key")
    answer: str = Field(..., description="Answer to cache")

class CortexCacheWarmupRequest(BaseModel):
    genesis_queries: Optional[List[str]] = Field(None, description="Queries to pre-warm the cache")

class CortexGuardianWakeupRequest(BaseModel):
    mode: str = Field("HOLISTIC", description="Synthesis mode")

class CortexCaptureSnapshotRequest(BaseModel):
    manifest_files: List[str] = Field(..., description="Files to include in snapshot")
    snapshot_type: str = Field("audit", description="'audit' or 'seal'")
    strategic_context: Optional[str] = Field(None, description="Context for snapshot")

class CortexLearningDebriefRequest(BaseModel):
    hours: int = Field(24, description="Lookback window in hours")

class ForgeQueryRequest(BaseModel):
    prompt: str = Field(..., description="Model prompt")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Max tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System persona prompt")

class CortexPersistSoulRequest(BaseModel):
    snapshot_path: str = Field(".agent/learning/learning_package_snapshot.md", description="Local path to seal")
    valence: float = Field(0.0, description="Moral/Emotional charge")
    uncertainty: float = Field(0.0, description="Logic confidence")
    is_full_sync: bool = Field(False, description="Sync entire learning directory if True")


# ============================================================================
# Helper Functions
# ============================================================================

def to_dict(obj: Any) -> Dict[str, Any]:
    #============================================
    # Function: to_dict
    # Purpose: Convert dataclass to dictionary recursively.
    # Args:
    #   obj: The dataclass object to convert
    # Returns: Dictionary representation
    #============================================
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
