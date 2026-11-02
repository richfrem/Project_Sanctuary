#!/usr/bin/env python3
"""
record_insight.py - Asymmetric Learning Persistence for Autonomous Agents
========================================================================

Purpose:
    Enforces the Asymmetric Persistence Rule: when an evolution attempt,
    test suite, or feature branch fails and code is rolled back, insights,
    edge cases, and friction logs are NEVER rolled back.

    Implements a two-phase architecture:
    1. Capture: Appends an atomic, immutable event to `.agent/learning/events.jsonl`.
    2. Promotion (Optional): Promotes curated insights into `context/memory.md`
       or `LEARNING/concepts/` / `LEARNING/topics/`.

Usage:
    python scripts/record_insight.py --topic "git_hooks" --type friction --insight "Pre-commit hook failed on zsh glob syntax"
    python scripts/record_insight.py --topic "attention_sinks" --type confirmed --insight "KV cache requires pin at position 0" --destination both
    python scripts/record_insight.py --topic "fast_boot" --type observed --insight "Read-only wakeup prevents git dirty states" --events-only
"""

import os
import sys
import json
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def append_event_log(root: Path, event: dict) -> Path:
    """Atomically append an immutable event to .agent/learning/events.jsonl."""
    events_file = root / ".agent" / "learning" / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(event) + "\n"

    # Append to file with immediate flush
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()

    return events_file


def record_to_context_memory(root: Path, topic: str, insight_type: str, text: str) -> None:
    """Idempotently append a curated bullet into context/memory.md."""
    memory_file = root / "context" / "memory.md"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bullet = f"- [{today}] [{insight_type.upper()}] {topic}: {text}\n"

    if memory_file.exists():
        content = memory_file.read_text(encoding="utf-8")
        if "## Active State & Epistemic Insights" not in content:
            content += "\n## Active State & Epistemic Insights\n"
        if text not in content:
            content += bullet
            memory_file.write_text(content, encoding="utf-8")
    else:
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        memory_file.write_text(f"# Active Memory\n\n## Active State & Epistemic Insights\n{bullet}", encoding="utf-8")


def record_to_learning_concepts(root: Path, topic: str, insight_type: str, text: str) -> Path:
    """Create or append an atomic concept note in LEARNING/concepts/."""
    safe_topic = topic.lower().replace(" ", "_").replace("-", "_")
    concept_file = root / "LEARNING" / "concepts" / f"{safe_topic}.md"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    status = "CONFIRMED" if insight_type.lower() == "confirmed" else "OBSERVED"

    if not concept_file.exists():
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        content = f"""# {topic.replace('_', ' ').title()}

**Status:** {status}  
**Source:** Autonomous Agent Session  
**Last Verified:** {today}  
**Tags:** ["{safe_topic}", "asymmetric_record", "{insight_type.lower()}"]

---

## 1. Overview & Recorded Insight

{text}

## 2. History & Epistemic Traces
- **{today}**: Recorded via asymmetric persistence ({insight_type.upper()}).
"""
        concept_file.write_text(content, encoding="utf-8")
    else:
        existing = concept_file.read_text(encoding="utf-8")
        if text not in existing:
            addition = f"\n- **{today}** ({insight_type.upper()}): {text}\n"
            if "## 2. History & Epistemic Traces" in existing:
                existing += addition
            else:
                existing += f"\n## 2. History & Epistemic Traces\n{addition}"
            concept_file.write_text(existing, encoding="utf-8")

    return concept_file


def main():
    parser = argparse.ArgumentParser(description="Asymmetric Learning Persistence for Autonomous Agents")
    parser.add_argument("--topic", required=True, help="Topic identifier (e.g. git_hooks, rlm_cache)")
    parser.add_argument("--type", choices=["friction", "confirmed", "hypothesis", "observed"], default="observed",
                        help="Insight epistemic classification")
    parser.add_argument("--insight", required=True, help="Detailed insight or friction description")
    parser.add_argument("--destination", choices=["events", "concepts", "memory", "both", "all"], default="both",
                        help="Destination for the insight promotion")
    parser.add_argument("--events-only", action="store_true", help="Only append to the immutable event log without markdown promotion")
    parser.add_argument("--evidence", default="", help="Optional source file or evidence reference")
    parser.add_argument("--session-id", default=os.environ.get("CONVERSATION_ID", "local-session"), help="Session ID")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    root = get_project_root()
    event_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    event = {
        "event_id": event_id,
        "timestamp": timestamp,
        "session_id": args.session_id,
        "topic": args.topic,
        "type": args.type.upper(),
        "insight": args.insight,
        "evidence": args.evidence
    }

    # Step 1: Always capture to immutable events.jsonl
    events_path = append_event_log(root, event)

    destinations_recorded = [str(events_path.relative_to(root))]

    # Step 2: Promote if not events-only
    if not args.events_only:
        if args.destination in ["memory", "both", "all"]:
            record_to_context_memory(root, args.topic, args.type, args.insight)
            destinations_recorded.append("context/memory.md")

        if args.destination in ["concepts", "both", "all"]:
            concept_path = record_to_learning_concepts(root, args.topic, args.type, args.insight)
            destinations_recorded.append(str(concept_path.relative_to(root)))

    if args.format == "json":
        print(json.dumps({
            "status": "success",
            "event_id": event_id,
            "timestamp": timestamp,
            "topic": args.topic,
            "destinations": destinations_recorded
        }, indent=2))
    else:
        print(f"Captured event '{event_id}' ({args.type.upper()}) into: {', '.join(destinations_recorded)}")


if __name__ == "__main__":
    main()
