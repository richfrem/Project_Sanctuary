# council_orchestrator/orchestrator/packets/emitter.py
# Packet emission utilities for JSONL and stdout streaming

import os
import sys
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import CouncilRoundPacket

def emit_packet(packet: "CouncilRoundPacket", jsonl_dir: str, stream_stdout: bool, schema_path: str = None):
    """Emit round packet to JSONL file and optionally stdout."""
    from .schema import validate_packet

    payload = asdict(packet)
    line = json.dumps(payload, ensure_ascii=False, default=str)

    # Validate against schema if available
    if not validate_packet(packet, schema_path):
        return False

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

    return True