#!/usr/bin/env python3
"""
RAG Cortex Models
=====================================

Purpose:
    Data definitions and Pydantic models for the RAG Cortex.
    Serves as the Single Source of Truth for MCP schemas.

Layer: Data (DTOs)

Key Models:
    - DocumentSample / CollectionStats / StatsResponse
    - IngestIncrementalRequest / IngestIncrementalResponse
    - CacheGetResponse / CacheSetResponse / CacheWarmupResponse
    - Opinion / DispositionParameters / HistoryPoint

    # Pydantic Models (MCP Requests)
    - CortexIngestFullRequest
    - CortexQueryRequest
    - CortexIngestIncrementalRequest
    - CortexCacheGetRequest
    - CortexCacheSetRequest
    - CortexCacheWarmupRequest
    - ForgeQueryRequest

Functions:
    - to_dict(obj): Recursive dataclass converter

Related:
    - mcp_servers/rag_cortex/operations.py
    - mcp_servers/rag_cortex/server.py
"""

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





# ============================================================================



# ============================================================================
# Opinion Models (ADR 091 - The Synaptic Phase)
# ============================================================================

@dataclass
class DispositionParameters:
    #============================================
    # Model: DispositionParameters
    # Purpose: Behavioral parameters from HINDSIGHT/CARA.
    #============================================
    skepticism: float
    literalism: float
    empathy: float = 0.5  # Default

@dataclass
class HistoryPoint:
    #============================================
    # Model: HistoryPoint
    # Purpose: Tracking confidence trajectory over time.
    #============================================
    timestamp: str
    score: float
    delta_reason: str

@dataclass
class Opinion:
    #============================================
    # Model: Opinion
    # Purpose: Subjective belief node (Synaptic Phase).
    #============================================
    id: str
    statement: str
    confidence_score: float
    formation_source: str
    supporting_evidence_ids: List[str]
    history_trajectory: List[HistoryPoint]
    disposition_parameters: Optional[DispositionParameters] = None
    type: str = "opinion"  # Discriminator

# ============================================================================



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





class ForgeQueryRequest(BaseModel):
    prompt: str = Field(..., description="Model prompt")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Max tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System persona prompt")




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
