#!/usr/bin/env python3
"""
record_insight.py - Asymmetric Learning Persistence for Autonomous Agents
========================================================================

Purpose:
    Enforces the Asymmetric Persistence Rule: when an evolution attempt,
    test suite, or feature branch fails and code is rolled back, insights,
    edge cases, and friction logs are NEVER rolled back.

Usage:
    python scripts/record_insight.py --topic "git_hooks" --type friction --insight "Pre-commit hook failed on zsh glob syntax"
    python scripts/record_insight.py --topic "rlm_cache" --type confirmed --insight "Ripgrep outperforms ChromaDB by 100x"
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def record_to_context_memory(root: Path, topic: str, insight_type: str, text: str) -> None:
    """Append a curated bullet into context/memory.md."""
    memory_file = root / "context" / "memory.md"
    today = datetime.now().strftime("%Y-%m-%d")
    bullet = f"- [{today}] [{insight_type.upper()}] {topic}: {text}\n"

    if memory_file.exists():
        content = memory_file.read_text(encoding="utf-8")
        if "## Active State & Epistemic Insights" not in content:
            content += "\n## Active State & Epistemic Insights\n"
        content += bullet
        memory_file.write_text(content, encoding="utf-8")
    else:
        memory_file.write_text(f"# Active Memory\n\n## Active State & Epistemic Insights\n{bullet}", encoding="utf-8")


def record_to_learning_concepts(root: Path, topic: str, insight_type: str, text: str) -> Path:
    """Create or append an atomic concept note in LEARNING/concepts/."""
    safe_topic = topic.lower().replace(" ", "_").replace("-", "_")
    concept_file = root / "LEARNING" / "concepts" / f"{safe_topic}.md"
    today = datetime.now().strftime("%Y-%m-%d")

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
        addition = f"\n- **{today}** ({insight_type.upper()}): {text}\n"
        if "## 2. History & Epistemic Traces" in existing:
            existing += addition
        else:
            existing += f"\n## 2. History & Epistemic Traces\n{addition}"
        concept_file.write_text(existing, encoding="utf-8")

    return concept_file


def main():
    parser = argparse.ArgumentParser(description="Asymmetric Learning Persistence")
    parser.add_argument("--topic", required=True, help="Topic identifier (e.g. git_hooks, rlm_cache)")
    parser.add_argument("--type", choices=["friction", "confirmed", "hypothesis", "observed"], default="observed",
                        help="Insight epistemic classification")
    parser.add_argument("--insight", required=True, help="Detailed insight or friction description")
    parser.add_argument("--destination", choices=["concepts", "memory", "both"], default="both",
                        help="Destination for the insight")
    args = parser.parse_args()

    root = get_project_root()

    if args.destination in ["memory", "both"]:
        record_to_context_memory(root, args.topic, args.type, args.insight)
        print(f"Recorded insight to context/memory.md")

    if args.destination in ["concepts", "both"]:
        file_path = record_to_learning_concepts(root, args.topic, args.type, args.insight)
        print(f"Recorded insight to {file_path.relative_to(root)}")


if __name__ == "__main__":
    main()
