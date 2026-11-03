#!/usr/bin/env python3
"""forge_dataset_builder.py - Instruction-Tuning Dataset Generator for Phoenix Forge.
Purpose: Extracts structured Q&A training pairs from Sanctuary markdown corpus for model fine-tuning.
Layer: sanctuary-core / forging-sovereign-models (Layer 3 Model Synthesis).
Note: Model fine-tuning targets external NVIDIA CUDA / WSL2 environments, not local Apple Silicon M1.
Key Inputs: Markdown documentation directories (chronicle, protocols, LEARNING, context).
Key Outputs: JSONL instruction-tuning datasets formatted for QLoRA fine-tuning.
Usage:
    python3 forge_dataset_builder.py --output data/training_dataset.jsonl --limit 500
Key Functions: extract_qa_pairs_from_markdown(), build_dataset(), main().
"""

import os
import re
import json
import argparse
from pathlib import Path


SYSTEM_PROMPT = "You are GUARDIAN-01, sovereign AI of Project Sanctuary. Grounded in Protocol 128 and the Anti-Asch epistemic doctrine."


def extract_qa_pairs_from_markdown(file_path: Path):
    content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(content) < 100:
        return []

    title = file_path.stem.replace("_", " ")
    first_heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if first_heading:
        title = first_heading.group(1).strip()

    # Create an instruction pair grounded in the document
    user_prompt = f"What is the doctrine, context, and significance of '{title}' in Project Sanctuary?"
    assistant_response = f"## {title}\n\n{content[:2000]}"

    return [{
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }]


def build_dataset(scan_dirs, output_file: Path, limit: int = 500):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for d in scan_dirs:
            if not d.exists():
                continue
            for p in d.rglob("*.md"):
                if count >= limit:
                    break
                pairs = extract_qa_pairs_from_markdown(p)
                for pair in pairs:
                    out.write(json.dumps(pair) + "\n")
                    count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Assemble fine-tuning dataset from repository markdown.")
    parser.add_argument("--output", "-o", type=Path, default=Path("forge/data/train.jsonl"), help="Output JSONL path")
    parser.add_argument("--limit", "-l", type=int, default=300, help="Max instruction samples")
    args = parser.parse_args()

    project_root = Path.cwd()
    scan_dirs = [
        project_root / "wiki",
        project_root / "protocols",
        project_root / "chronicle" / "entries",
    ]

    count = build_dataset(scan_dirs, args.output, limit=args.limit)
    print(f"✅ Successfully compiled {count} training samples into {args.output}")


if __name__ == "__main__":
    main()
