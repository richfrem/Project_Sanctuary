# council_orchestrator/packets.py
# Packet schema and utilities for the orchestrator

import os
import sys
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

def emit_packet(packet: CouncilRoundPacket, jsonl_dir: str, stream_stdout: bool, schema_path: str = None):
    """Emit round packet to JSONL file and optionally stdout."""
    payload = asdict(packet)
    line = json.dumps(payload, ensure_ascii=False, default=str)

    # Validate against schema if available
    if schema_path and os.path.exists(schema_path):
        try:
            import jsonschema
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            jsonschema.validate(instance=payload, schema=schema)
        except ImportError:
            pass  # Schema validation not available
        except Exception as e:
            print(f"[SCHEMA VALIDATION ERROR] {e}")

    # File persistence
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)
        jsonl_path = os.path.join(jsonl_dir, f"{packet.session_id}", f"round_{packet.round_id}.jsonl")
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Stdout streaming
    if stream_stdout:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()