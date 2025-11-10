# council_orchestrator/orchestrator/packets/schema.py
# Packet schema and validation utilities

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

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
    novelty: Dict[str, Any]
    memory_directive: Dict[str, str]
    cost: Dict[str, Any]
    errors: List[str]
    schema_version: str = "1.0.0"

# --- ROUND PACKET UTILITIES ---
def seed_for(session_id: str, round_id: int, member_id: str) -> int:
    """Generate deterministic seed for reproducibility."""
    try:
        import xxhash
        return xxhash.xxh64_intdigest(f"{session_id}:{round_id}:{member_id}") & 0x7fffffff
    except ImportError:
        # Fallback to hashlib if xxhash not available
        hash_obj = hashlib.md5(f"{session_id}:{round_id}:{member_id}".encode())
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