#!/usr/bin/env python3
"""
Scan a directory of Python files and extract metadata:
name, lines, first_docstring_line, has_type_hints, has_test_block.
Outputs JSON to stdout or a file.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def analyze_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = len(text.splitlines())
    try:
        tree = ast.parse(text)
    except Exception:
        tree = ast.Module(body=[])  # fallback
    doc = ast.get_docstring(tree)
    first_doc_line: Optional[str] = None
    if doc:
        for ln in doc.splitlines():
            s = ln.strip()
            if s:
                first_doc_line = s
                break

    has_type_hints = False
    has_test_block = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # args annotations
            args = getattr(node, "args", None)
            if args:
                for a in getattr(args, "args", []) + getattr(args, "kwonlyargs", []):
                    if getattr(a, "annotation", None) is not None:
                        has_type_hints = True
                        break
                if getattr(args, "vararg", None) and getattr(args.vararg, "annotation", None) is not None:
                    has_type_hints = True
                if getattr(args, "kwarg", None) and getattr(args.kwarg, "annotation", None) is not None:
                    has_type_hints = True
            if getattr(node, "returns", None) is not None:
                has_type_hints = True
            if node.name.startswith("test_"):
                has_test_block = True
        if isinstance(node, ast.AnnAssign):
            has_type_hints = True
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in ("unittest", "pytest"):
                    has_test_block = True
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in ("unittest", "pytest"):
                has_test_block = True
        if isinstance(node, ast.If):
            test = node.test
            # check for if __name__ == "__main__"
            if isinstance(test, ast.Compare):
                left = test.left
                comps = test.comparators if hasattr(test, "comparators") else []
                if isinstance(left, ast.Name) and left.id == "__name__":
                    for c in comps:
                        if isinstance(c, (ast.Constant, ast.Str)) and getattr(c, "s", getattr(c, "value", None)) == "__main__":
                            has_test_block = True
    return {
        "name": path.name,
        "lines": lines,
        "first_docstring_line": first_doc_line,
        "has_type_hints": bool(has_type_hints),
        "has_test_block": bool(has_test_block),
    }


def scan_toolbox(toolbox_path: str) -> List[Dict[str, Any]]:
    p = Path(toolbox_path)
    files: List[Path] = sorted(p.rglob("*.py"))
    results: List[Dict[str, Any]] = []
    for f in files:
        # skip __pycache__ paths just in case
        if "__pycache__" in f.parts:
            continue
        results.append(analyze_file(f))
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Scan python toolbox and emit JSON metadata.")
    parser.add_argument("--toolbox-path", "-p", default=".", help="Path to scan for .py files")
    parser.add_argument("--output", "-o", default="-", help="Output file path (default stdout). Use '-' for stdout.")
    args = parser.parse_args(argv)
    data = scan_toolbox(args.toolbox_path)
    out = json.dumps(data, indent=2, ensure_ascii=False)
    if args.output == "-" or args.output.strip() == "":
        print(out)
    else:
        Path(args.output).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    # Basic self-test assertions (no file creation)
    res = scan_toolbox(".")
    assert isinstance(res, list), "scan should return a list"
    for item in res:
        assert "name" in item and "lines" in item and "has_type_hints" in item and "has_test_block" in item
    # If invoked directly, run main()
    main()

