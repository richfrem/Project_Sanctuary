#!/usr/bin/env python3
"""
query_graph.py - Instant Wikilink & Associative Graph Traverser
=============================================================

Purpose:
    Traverses inbound and outbound wikilinks ([[concept]]) and standard
    relative Markdown links across wiki/, protocols/, and LEARNING/
    in milliseconds using only Python standard library.

Usage:
    python scripts/query_graph.py --concept sovereign_memory_engine
    python scripts/query_graph.py --file wiki/concepts/anti_asch_engine.md
"""

import re
import sys
import argparse
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def extract_links(content: str) -> list[str]:
    """Find both [[wikilinks]] and [text](relative_links.md)."""
    links = []
    # Match [[concept]]
    for m in re.finditer(r"\[\[([^\]]+)\]\]", content):
        links.append(m.group(1).strip())
    # Match [label](path)
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+\.md)\)", content):
        target = m.group(2).split("#")[0].strip()
        links.append(Path(target).stem)
    return links


def build_graph(root: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Scan markdown files and build forward and backward link indices."""
    outbound: dict[str, set[str]] = {}
    inbound: dict[str, set[str]] = {}

    target_dirs = ["wiki", "protocols", "LEARNING"]

    for d in target_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("**/*.md"):
            if "ARCHIVE" in md_file.parts:
                continue
            stem = md_file.stem.lower()
            try:
                content = md_file.read_text(encoding="utf-8")
                links = extract_links(content)
                if stem not in outbound:
                    outbound[stem] = set()
                for target in links:
                    target_stem = target.lower()
                    if target_stem != stem:
                        outbound[stem].add(target_stem)
                        if target_stem not in inbound:
                            inbound[target_stem] = set()
                        inbound[target_stem].add(stem)
            except Exception:
                continue

    return outbound, inbound


def query_concept(root: Path, concept_name: str) -> None:
    concept = concept_name.lower().replace(".md", "")
    outbound, inbound = build_graph(root)

    print(f"=== Associative Graph: '{concept}' ===")
    
    out_links = sorted(outbound.get(concept, set()))
    print(f"\nOutbound References ({len(out_links)}):")
    if out_links:
        for link in out_links:
            print(f"  -> {link}")
    else:
        print("  (None)")

    in_links = sorted(inbound.get(concept, set()))
    print(f"\nInbound Backlinks ({len(in_links)}):")
    if in_links:
        for link in in_links:
            print(f"  <- {link}")
    else:
        print("  (None)")


def main():
    parser = argparse.ArgumentParser(description="Associative Concept Graph Traverser")
    parser.add_argument("--concept", type=str, help="Concept identifier or file stem")
    parser.add_argument("--file", type=str, help="Markdown file path to inspect")
    args = parser.parse_args()

    root = get_project_root()
    concept = args.concept
    if not concept and args.file:
        concept = Path(args.file).stem

    if not concept:
        print("Error: Specify either --concept or --file.", file=sys.stderr)
        sys.exit(1)

    query_concept(root, concept)


if __name__ == "__main__":
    main()
