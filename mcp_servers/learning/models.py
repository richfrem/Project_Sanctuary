
#============================================
# mcp_servers/learning/models.py
# Purpose: Pydantic/Dataclass models for Learning Loop operations.
# Protocol: 128 (Cognitive Continuity)
#============================================

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# ============================================================================
# Cache Operation Models (Protocol 114 - Guardian Wakeup)
# ============================================================================

@dataclass
class CacheGetResponse:
    cache_hit: bool
    answer: Optional[str]
    query_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None

@dataclass
class CacheSetResponse:
    cache_key: str
    stored: bool
    status: str
    error: Optional[str] = None

@dataclass
class CacheWarmupResponse:
    queries_cached: int
    cache_hits: int
    cache_misses: int
    total_time_ms: float
    status: str
    error: Optional[str] = None

@dataclass
class GuardianWakeupResponse:
    digest_path: str
    bundles_loaded: List[str] = None
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_ms: float = 0.0
    status: str = "success"
    error: Optional[str] = None

# ============================================================================
# Capture Snapshot Models (Protocol 128 v3.5)
# ============================================================================

@dataclass
class CaptureSnapshotRequest:
    manifest_files: List[str]
    snapshot_type: str = "audit"  # "audit" or "seal"
    strategic_context: Optional[str] = None

@dataclass
class CaptureSnapshotResponse:
    snapshot_path: str
    manifest_verified: bool
    git_diff_context: str
    snapshot_type: str
    status: str
    total_files: int = 0
    total_bytes: int = 0
    error: Optional[str] = None

# ============================================================================
# Opinion Models (ADR 091 - The Synaptic Phase)
# ============================================================================

@dataclass
class DispositionParameters:
    skepticism: float
    literalism: float
    empathy: float = 0.5

@dataclass
class HistoryPoint:
    timestamp: str
    score: float
    delta_reason: str

@dataclass
class Opinion:
    id: str
    statement: str
    confidence_score: float
    formation_source: str
    supporting_evidence_ids: List[str]
    history_trajectory: List[HistoryPoint]
    disposition_parameters: Optional[DispositionParameters] = None
    type: str = "opinion"

# ============================================================================
# Persist Soul Models (ADR 079 - Johnny Appleseed)
# ============================================================================

@dataclass
class PersistSoulRequest:
    snapshot_path: str = ".agent/learning/learning_package_snapshot.md"
    valence: float = 0.0
    uncertainty: float = 0.0
    is_full_sync: bool = False

@dataclass
class PersistSoulResponse:
    status: Optional[str] = None
    repo_url: Optional[str] = None
    snapshot_name: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# FastMCP Request Models
# ============================================================================

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

class CortexPersistSoulRequest(BaseModel):
    snapshot_path: str = Field(".agent/learning/learning_package_snapshot.md", description="Local path to seal")
    valence: float = Field(0.0, description="Moral/Emotional charge")
    uncertainty: float = Field(0.0, description="Logic confidence")
    is_full_sync: bool = Field(False, description="Sync entire learning directory if True")

@dataclass
class GuardianSnapshotResponse:
    status: str
    snapshot_path: str
    total_files: int = 0
    total_bytes: int = 0
    error: Optional[str] = None

class CortexGuardianSnapshotRequest(BaseModel):
    strategic_context: Optional[str] = Field(None, description="Context for guardian snapshot")

# ============================================================================
# Helper Functions
# ============================================================================

def to_dict(obj: Any) -> Dict[str, Any]:
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
