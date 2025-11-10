# council_orchestrator/events.py
"""
Event logging and management system for orchestrator observability.
Handles structured JSON event logging, aggregation, and round analysis.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List


class EventManager:
    """
    Manages structured event logging and aggregation for orchestrator observability.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.event_log_path = project_root / "logs" / "events.jsonl"
        self.run_id = f"run_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        self.event_buffer = []

    def setup_event_logging(self):
        """Initialize structured JSON event logging system for observability."""
        print(f"[+] Event logging initialized - Run ID: {self.run_id}")

    def emit_event(self, event_type: str, **kwargs):
        """Emit a structured JSON event to the event log.

        Event Schema:
        - ts: ISO timestamp
        - run_id: Unique run identifier
        - event_type: member_response|round_complete|task_start|task_complete|error
        - round: Round number (for member_response/round_complete)
        - member_id: Agent role identifier
        - role: Agent role name
        - status: success|error|timeout
        - latency_ms: Response time in milliseconds
        - tokens_in: Input tokens used
        - tokens_out: Output tokens generated
        - result_type: analysis|proposal|critique|consensus
        - score: Quality/confidence score (0.0-1.0)
        - vote: Agent's vote/decision
        - novelty: fast|medium|slow (memory placement hint)
        - reasons: List of reasoning factors
        - citations: List of referenced content
        - errors: List of error messages
        - content_ref: Reference to stored content
        """
        event = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event_type": event_type,
            **kwargs
        }

        # Write to buffer and flush to file
        self.event_buffer.append(event)
        self._flush_events()

        # Log to console for real-time monitoring
        if event_type == "member_response":
            status_emoji = "✅" if kwargs.get("status") == "success" else "❌"
            print(f"{status_emoji} [{kwargs.get('role', 'unknown')}] Round {kwargs.get('round', '?')} - {kwargs.get('latency_ms', 0)}ms", flush=True)

    def _flush_events(self):
        """Flush buffered events to JSONL file."""
        try:
            with open(self.event_log_path, 'a', encoding='utf-8') as f:
                for event in self.event_buffer:
                    f.write(json.dumps(event, default=str) + '\n')
            self.event_buffer.clear()
        except Exception as e:
            print(f"[EVENT LOG ERROR] Failed to write events: {e}")