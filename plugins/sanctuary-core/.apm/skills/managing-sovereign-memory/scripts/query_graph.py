#!/usr/bin/env python3
"""
query_graph.py - Instant Wikilink & Associative Graph Traverser
=============================================================

Purpose:
    Traverses inbound and outbound wikilinks ([[concept]]) and standard
    relative Markdown links across LEARNING/ and protocols/ in milliseconds
    using only Python standard library.

Usage:
    python scripts/query_graph.py --target sovereign_memory_engine
    python scripts/query_graph.py --target sovereign_memory_engine --format json
    python scripts/query_graph.py --file LEARNING/concepts/anti_asch_engine.md
"""

import re
import sys
import json
import argparse
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def extract_links(content: str) -> list[str]:
    """Extract and normalize link targets from wikilinks and standard Markdown links."""
    links = []
    # Match [[concept]], [[concept|alias]], [[concept#heading|alias]]
    for m in re.finditer(r"\[\[([^\]]+)\]\]", content):
        raw = m.group(1).strip()
        # Strip pipe alias first: concept#heading|alias -> concept#heading
        target = raw.split("|")[0].strip()
        # Strip section/anchor: concept#heading -> concept
        target = target.split("#")[0].strip()
        if target:
            links.append(Path(target).stem.lower())

    # Match [label](relative_path.md) or [label](relative_path.md#heading)
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+\.md(?:#[^)]*)?)\)", content):
        raw = m.group(2).strip()
        target = raw.split("#")[0].strip()
        if target and not target.startswith("http://") and not target.startswith("https://"):
            links.append(Path(target).stem.lower())

    return links


def build_graph(root: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Scan markdown files and build forward and backward link indices with path mapping."""
    outbound: dict[str, set[str]] = {}
    inbound: dict[str, set[str]] = {}
    path_map: dict[str, str] = {}

    target_dirs = ["protocols", "LEARNING"]

    for d in target_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("**/*.md"):
            if "ARCHIVE" in md_file.parts or ".agent" in md_file.parts:
                continue
            stem = md_file.stem.lower()
            rel_path = str(md_file.relative_to(root))
            path_map[stem] = rel_path

            try:
                content = md_file.read_text(encoding="utf-8")
                links = extract_links(content)
                if stem not in outbound:
                    outbound[stem] = set()
                for target_stem in links:
                    if target_stem != stem:
                        outbound[stem].add(target_stem)
                        if target_stem not in inbound:
                            inbound[target_stem] = set()
                        inbound[target_stem].add(stem)
            except Exception:
                continue

    return outbound, inbound, path_map


def query_concept(root: Path, concept_name: str, output_format: str = "text") -> None:
    concept = concept_name.lower().replace(".md", "")
    outbound, inbound, path_map = build_graph(root)

    out_links = sorted(outbound.get(concept, set()))
    in_links = sorted(inbound.get(concept, set()))
    rel_path = path_map.get(concept)

    if output_format == "json":
        data = {
            "concept": concept,
            "path": rel_path,
            "outbound_count": len(out_links),
            "outbound": [{"target": link, "path": path_map.get(link)} for link in out_links],
            "inbound_count": len(in_links),
            "inbound": [{"source": link, "path": path_map.get(link)} for link in in_links]
        }
        print(json.dumps(data, indent=2))
        return

    print(f"=== Associative Graph: '{concept}' ===")
    if rel_path:
        print(f"Source Document: {rel_path}")
    print(f"\nOutbound References ({len(out_links)}):")
    if out_links:
        for link in out_links:
            target_file = f" ({path_map[link]})" if link in path_map else ""
            print(f"  -> {link}{target_file}")
    else:
        print("  (None)")

    print(f"\nInbound Backlinks ({len(in_links)}):")
    if in_links:
        for link in in_links:
            source_file = f" ({path_map[link]})" if link in path_map else ""
            print(f"  <- {link}{source_file}")
    else:
        print("  (None)")


def main():
    parser = argparse.ArgumentParser(description="Associative Concept Graph Traverser")
    parser.add_argument("--target", "-t", type=str, help="Concept identifier or file stem to query")
    parser.add_argument("--concept", "-c", type=str, help="Alias for --target")
    parser.add_argument("--file", "-f", type=str, help="Markdown file path to inspect")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    concept = args.target or args.concept
    if not concept and args.file:
        concept = Path(args.file).stem

    if not concept:
        print("Error: Specify either --target, --concept, or --file.", file=sys.stderr)
        sys.exit(1)

    root = get_project_root()
    query_concept(root, concept, output_format=args.format)


if __name__ == "__main__":
    main()
