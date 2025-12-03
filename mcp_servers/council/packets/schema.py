# council_orchestrator/orchestrator/packets/schema.py
# Packet schema and validation utilities

import json
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

# --- Phase 2 additions ---
@dataclass
class MemoryDirectiveField:
    tier: str                      # "fast" | "medium" | "slow"
    justification: str

@dataclass
class NoveltyField:
    is_novel: bool
    signal: str                    # "none"|"low"|"medium"|"high"
    basis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConflictField:
    conflicts_with: List[str] = field(default_factory=list)
    basis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalField:
    structured_query: Dict[str, Any] = field(default_factory=dict)
    parent_docs: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_latency_ms: int = 0
    plan_latency_ms: int = 0
    analyze_latency_ms: int = 0
    emit_latency_ms: int = 0

# --- COUNCIL ROUND PACKET SCHEMA ---
@dataclass
class CouncilRoundPacket:
    timestamp: str
    session_id: str
    round_id: int
    member_id: str
    engine: str
    seed: int
    prompt_hash: str
    inputs: Dict[str, Any]
    decision: str
    rationale: str
    confidence: float
    citations: List[Dict[str, str]]
    rag: Dict[str, Any]
    cag: Dict[str, Any]
    novelty: NoveltyField = field(default_factory=lambda: NoveltyField(False,"none",{}))
    memory_directive: MemoryDirectiveField = field(default_factory=lambda: MemoryDirectiveField("fast","initial default"))
    cost: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    schema_version: str = "1.0.0"
    # --- Phase 2 additions ---
    retrieval: RetrievalField = field(default_factory=RetrievalField)
    conflict: ConflictField = field(default_factory=ConflictField)
    seed_chain: Dict[str, Any] = field(default_factory=dict)  # Provenance for deterministic replay

# --- ROUND PACKET UTILITIES ---
def seed_for(session_id: str, round_id: int, member_id: str, prompt_hash: str = None) -> int:
    """Generate deterministic seed for reproducibility."""
    seed_input = f"{session_id}:{round_id}:{member_id}"
    if prompt_hash:
        seed_input += f":{prompt_hash}"

    try:
        import xxhash
        return xxhash.xxh64_intdigest(seed_input) & 0x7fffffff
    except ImportError:
        # Fallback to SHA256 if xxhash not available
        hash_obj = hashlib.sha256(seed_input.encode())
        return int(hash_obj.hexdigest(), 16) & 0x7fffffff

def prompt_hash(text: str) -> str:
    """Generate hash for prompt content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def validate_packet(packet: CouncilRoundPacket, schema_path: str = None) -> bool:
    """Validate packet against JSON schema if available."""
    if not schema_path:
        return True

    try:
        import jsonschema
        payload = asdict(packet)
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        jsonschema.validate(instance=payload, schema=schema)
        return True
    except ImportError:
        return True  # Schema validation not available
    except Exception as e:
        print(f"[SCHEMA VALIDATION ERROR] {e}")
        return False