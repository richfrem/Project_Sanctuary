#!/usr/bin/env python3
"""
bootstrap_briefing_packet.py
Generates a dynamic briefing_packet.json for synchronized Council deliberations.

Steps:
1. Load the last 2 entries from Living_Chronicle.md (temporal anchors).
2. Load the last 2 directives from WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/.
3. Construct briefing_packet.json with metadata, anchors, summaries, and current task.
4. Save to WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

# --- CONFIG ---
CHRONICLE_PATH = Path("../00_CHRONICLE/Living_Chronicle.md")
DIRECTIVES_DIR = Path("../WORK_IN_PROGRESS/COUNCIL_DIRECTIVES")
OUTPUT_PATH = Path("../WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json")

def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_latest_chronicle_entries(n=2):
    """Parse the last n entries from Living_Chronicle.md."""
    if not CHRONICLE_PATH.exists():
        raise FileNotFoundError(f"{CHRONICLE_PATH} not found")

    lines = CHRONICLE_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    current_entry = []

    for line in lines:
        if line.startswith("Entry "):
            if current_entry:
                entries.append("\n".join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)

    if current_entry:
        entries.append("\n".join(current_entry))

    latest = entries[-n:]
    anchors = []
    for entry in latest:
        first_line = entry.splitlines()[0]
        title = first_line.strip()
        checksum = sha256_of_text(entry)
        anchors.append({"title": title, "checksum": checksum})
    return anchors

def get_latest_directives(n=2):
    """Fetch latest n directive summaries from WORK_IN_PROGRESS/COUNCIL_DIRECTIVES."""
    if not DIRECTIVES_DIR.exists():
        return []

    files = sorted(DIRECTIVES_DIR.glob("directive_*.md"), key=os.path.getmtime, reverse=True)
    directives = []
    for f in files[:n]:
        content = f.read_text(encoding="utf-8").splitlines()
        summary = content[1].strip() if len(content) > 1 else "Summary unavailable."
        directives.append({"directive_id": f.stem, "summary": summary})
    return directives

def get_current_command():
    """Read the live command.json if it exists."""
    COMMAND_PATH = Path("council_orchestrator/command.json")
    if COMMAND_PATH.exists():
        try:
            return json.loads(COMMAND_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[!] Error reading command.json: {e}")
            return None
    return None

def main():
    timestamp = datetime.utcnow().isoformat() + "Z"
    packet = {
        "metadata": {
            "packet_id": f"briefing_{timestamp.replace(':','-')}",
            "timestamp": timestamp,
            "generated_by": "bootstrap_briefing_packet.py",
            "protocols": ["Protocol 94", "Protocol 95", "Prometheus v9.3"]
        },
        "temporal_anchors": get_latest_chronicle_entries(2),
        "prior_directives_summary": get_latest_directives(2),
        "current_task": get_current_command() or {
            "directive_id": "directive_003_council_memory_sync",
            "description": "Establish Council Memory Synchronization Protocol."
        },
        "shared_context": {
            "notes": [
                "All Cortex queries must use standardized schema (see cortex_query_schema.json).",
                "Continuity must be verified against Chronicle anchors before major decisions.",
                "Logs of this briefing cycle should be stored in WORK_IN_PROGRESS/council_memory_sync/briefing_logs_<timestamp>.md"
            ]
        }
    }

    # MAP-94: Calculate attestation hash
    packet_for_hashing = packet.copy()
    del packet_for_hashing["metadata"]
    canonical_string = json.dumps(packet_for_hashing, sort_keys=True, separators=(',', ':'))
    attestation_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    packet["metadata"]["attestation_hash"] = attestation_hash

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    print(f"[+] briefing_packet.json generated at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()