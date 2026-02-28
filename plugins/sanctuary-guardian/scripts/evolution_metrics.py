#!/usr/bin/env python3
"""
evolution_metrics.py — Evolutionary Self-Improvement Metrics (Protocol 131)
===========================================================================

Purpose:
    Computes proxy fitness metrics (Depth, Scope) for the Map-Elites archive.
    Standalone CLI script — no mcp_servers dependencies.

Usage:
    python3 evolution_metrics.py fitness --file docs/my-document.md
    python3 evolution_metrics.py depth --file README.md
    python3 evolution_metrics.py scope --content "Some text content..."

Layer: Plugin Script (guardian-onboarding)
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict


def measure_depth(content: str) -> float:
    """
    Computes 'Depth' score (0.0 - 5.0) based on citation density and technical complexity.
    """
    if not content or not content.strip():
        return 0.0

    words = content.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # 1. Citation Density
    links = len(re.findall(r'\[.*?\]\(http.*?\)', content))
    citation_density = (links / word_count) * 100

    # 2. Avg Word Length (Simple complexity proxy)
    avg_len = sum(len(w) for w in words) / word_count

    # Heuristic scoring
    score = 0.0

    # Citation bonus (capped at 2.5)
    score += min(2.5, citation_density * 2.0)

    # Complexity bonus (capped at 2.5)
    # Assume avg length 4.5 is standard, 7 is technical
    complexity_bonus = max(0, (avg_len - 4.5))
    score += min(2.5, complexity_bonus)

    return float(round(score, 2))


def measure_scope(content: str) -> float:
    """
    Computes 'Scope' score (0.0 - 5.0) based on file touch width.
    """
    if not content or not content.strip():
        return 0.0

    # Extract file paths mentioned in content
    file_refs = set(re.findall(r'`([^`]+\.[a-zA-Z0-9]+)`', content))
    # Also look for [link](path) — Exclude external http/https and anchors
    raw_links = re.findall(r'\]\((.*?)\)', content)
    link_refs = {link for link in raw_links if not link.strip().startswith(('http', 'https', '#'))}

    all_refs = file_refs.union(link_refs)
    unique_files = len(all_refs)

    if unique_files == 0:
        return 0.0

    # Extract domains (top-level dirs)
    domains = set()
    for ref in all_refs:
        parts = ref.split('/')
        if len(parts) > 1:
            domains.add(parts[0])
        else:
            domains.add("root")

    # Heuristic Scoring
    score = 0.0

    # File count bonus (capped at 2.5) — 10 files = max score
    score += min(2.5, (unique_files / 10) * 2.5)

    # Domain bonus — 1 domain = narrow (0.5), 3+ domains = broad (2.5)
    domain_count = len(domains)
    score += min(2.5, (domain_count / 4) * 2.5)

    return float(round(score, 2))


def calculate_fitness(content: str) -> Dict[str, float]:
    """
    Calculate full fitness vector for an individual.
    """
    return {
        "depth": measure_depth(content),
        "scope": measure_scope(content)
    }


def main():
    parser = argparse.ArgumentParser(description="Evolution Metrics (Protocol 131)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fitness
    fit_parser = subparsers.add_parser("fitness", help="Calculate full fitness vector")
    fit_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    fit_parser.add_argument("--file", help="Read content from file")

    # depth
    depth_parser = subparsers.add_parser("depth", help="Evaluate technical depth")
    depth_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    depth_parser.add_argument("--file", help="Read content from file")

    # scope
    scope_parser = subparsers.add_parser("scope", help="Evaluate architectural scope")
    scope_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    scope_parser.add_argument("--file", help="Read content from file")

    args = parser.parse_args()

    # Resolve content
    content = args.content
    if args.file:
        try:
            content = Path(args.file).read_text(encoding='utf-8')
        except Exception as e:
            print(json.dumps({"status": "error", "error": f"Could not read file: {e}"}))
            sys.exit(1)

    if not content:
        print(json.dumps({"status": "error", "error": "No content provided. Use --file or pass text."}))
        sys.exit(1)

    if args.command == "fitness":
        result = calculate_fitness(content)
        print(json.dumps(result, indent=2))
    elif args.command == "depth":
        score = measure_depth(content)
        print(json.dumps({"depth": score}))
    elif args.command == "scope":
        score = measure_scope(content)
        print(json.dumps({"scope": score}))


if __name__ == "__main__":
    main()
