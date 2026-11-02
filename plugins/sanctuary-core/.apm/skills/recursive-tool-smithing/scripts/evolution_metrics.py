"""
evolution_metrics.py
====================
Purpose:
    Computes proxy fitness metrics for toolbox scripts to guide the keep/discard
    decision beyond simple pass/fail. Adapts the Depth + Scope scoring from
    sanctuary-guardian's evolution_metrics.py for tool-smithing use.

    Depth  = documentation density + type-hint coverage (quality of the script)
    Scope  = capability breadth + test coverage (how much problem space it covers)
    Fitness = average of Depth and Scope, scored 0.0 - 5.0

Layer: Plugin Utility Script
Plugin: recursive-tool-smithing

Usage:
    python3 evolution_metrics.py fitness --file scripts/toolbox/pdf_parser.py
    python3 evolution_metrics.py depth   --file scripts/toolbox/pdf_parser.py
    python3 evolution_metrics.py scope   --file scripts/toolbox/pdf_parser.py
    python3 evolution_metrics.py --help

Output:
    JSON: {"depth": float, "scope": float, "fitness": float}

Key Functions:
    - measure_depth(source) -> float
    - measure_scope(source) -> float
    - calculate_fitness(source) -> dict[str, float]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


# Depth metric: documentation density + type-hint coverage.
def measure_depth(source: str) -> float:
    """Score documentation and type-hint quality (0.0 - 5.0).

    Scoring:
    - Has module docstring: +0.5
    - Docstring density: (functions_with_docstring / total_functions) * 2.0
    - Type-hint coverage: (annotated_args / total_args) * 2.0
    - Inline comment density: capped at 0.5

    Args:
        source: Python source code as string.

    Returns:
        Depth score between 0.0 and 5.0.
    """
    if not source.strip():
        return 0.0

    score = 0.0

    # Module docstring
    if source.strip().startswith('"""') or source.strip().startswith("'''"):
        score += 0.5

    # Parse AST for function analysis
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return min(score, 5.0)

    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if functions:
        # Docstring coverage
        with_docstring = sum(1 for f in functions if ast.get_docstring(f))
        score += (with_docstring / len(functions)) * 2.0

        # Type-hint coverage
        total_args = sum(len(f.args.args) for f in functions)
        annotated_args = sum(
            sum(1 for a in f.args.args if a.annotation is not None)
            for f in functions
        )
        if total_args > 0:
            score += (annotated_args / total_args) * 2.0
    else:
        # No functions but has docstring: give partial credit
        score += 0.5

    # Inline comment bonus (capped at 0.5)
    lines = source.splitlines()
    comment_lines = sum(1 for ln in lines if ln.strip().startswith("#"))
    comment_ratio = comment_lines / max(len(lines), 1)
    score += min(comment_ratio * 2.0, 0.5)

    return min(round(score, 2), 5.0)


# Scope metric: capability breadth + test coverage.
def measure_scope(source: str) -> float:
    """Score capability breadth and test coverage (0.0 - 5.0).

    Scoring:
    - Public function count: min(count / 3.0, 2.5) (more distinct capabilities)
    - Has __main__ test block: +1.0
    - Has assertion(s) in test block: +0.5
    - Has argparse CLI: +0.5
    - Has error handling (try/except): +0.5

    Args:
        source: Python source code as string.

    Returns:
        Scope score between 0.0 and 5.0.
    """
    if not source.strip():
        return 0.0

    score = 0.0

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.0

    # Public function count (breadth of capabilities)
    public_funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not n.name.startswith("_")
    ]
    score += min(len(public_funcs) / 3.0, 2.5)

    # Has __main__ block (test coverage)
    has_main = "__main__" in source
    if has_main:
        score += 1.0

    # Has assertions in __main__ (test quality)
    if has_main and "assert " in source:
        score += 0.5

    # Has argparse (CLI usability)
    if "argparse" in source:
        score += 0.5

    # Has try/except (error handling)
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
    if has_try:
        score += 0.5

    return min(round(score, 2), 5.0)


# Calculate composite fitness from a source file.
def calculate_fitness(source: str) -> dict[str, float]:
    """Compute depth, scope, and composite fitness for a script.

    Args:
        source: Python source code as string.

    Returns:
        Dict with keys: depth (0-5), scope (0-5), fitness (0-5).
    """
    depth = measure_depth(source)
    scope = measure_scope(source)
    fitness = round((depth + scope) / 2.0, 2)
    return {"depth": depth, "scope": scope, "fitness": fitness}


def main() -> None:
    """CLI entry point for evolution_metrics."""
    parser = argparse.ArgumentParser(description="Compute fitness metrics for a toolbox script.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ("fitness", "depth", "scope"):
        p = subparsers.add_parser(cmd, help=f"Compute {cmd} metric")
        p.add_argument("--file", required=True, help="Path to Python script")

    args = parser.parse_args()
    source = Path(args.file).read_text(encoding="utf-8")

    if args.command == "depth":
        print(json.dumps({"depth": measure_depth(source)}))
    elif args.command == "scope":
        print(json.dumps({"scope": measure_scope(source)}))
    else:
        print(json.dumps(calculate_fitness(source)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)

    # Self-test (no CLI args)
    sample_good = '''"""
Good script with full coverage.

Purpose: Demonstrate high fitness.
"""
from __future__ import annotations
import argparse


def process_data(items: list[str]) -> list[str]:
    """Process a list of strings.

    Args:
        items: Input list.

    Returns:
        Processed list.
    """
    try:
        return [x.strip() for x in items if x]
    except AttributeError as e:
        raise ValueError(f"Invalid input: {e}") from e


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+")
    args = parser.parse_args()
    print(process_data(args.input or []))


if __name__ == "__main__":
    result = process_data(["  hello  ", "world"])
    assert result == ["hello", "world"], f"Got {result}"
    print("evolution_metrics self-test: sample_good passed.")
    main()
'''

    sample_bare = "x = 1\n"

    good_metrics = calculate_fitness(sample_good)
    bare_metrics = calculate_fitness(sample_bare)

    assert good_metrics["depth"] > bare_metrics["depth"], "Good script should have higher depth"
    assert good_metrics["scope"] > bare_metrics["scope"], "Good script should have higher scope"
    assert good_metrics["fitness"] > 3.0, f"Expected fitness > 3.0, got {good_metrics['fitness']}"
    assert bare_metrics["fitness"] < 1.0, f"Expected bare fitness < 1.0, got {bare_metrics['fitness']}"

    print(f"evolution_metrics: good={good_metrics}, bare={bare_metrics}")
    print("evolution_metrics: all self-tests passed.")
