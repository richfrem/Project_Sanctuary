#!/usr/bin/env python3
"""
disclose.py - Progressive Disclosure CLI for Sovereign AI Memory
==============================================================

Purpose:
    Retrieves knowledge at specific levels of granularity (Tier 1: Index,
    Tier 2: Summary, Tier 3: Deep) to preserve the agent's context window.

Usage Examples:
    python scripts/disclose.py --level index --target wiki/
    python scripts/disclose.py --level summary --file wiki/concepts/sovereign_memory_engine.md
    python scripts/disclose.py --level deep --file protocols/128_Hardened_Learning_Loop_v3.0.md
"""

import sys
import argparse
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def extract_summary(file_path: Path, max_lines: int = 30) -> str:
    """Extract frontmatter and executive summary from a markdown document."""
    if not file_path.exists():
        return f"Error: File '{file_path}' not found."
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = content.splitlines()
    summary_lines = []
    in_frontmatter = False
    frontmatter_closed = False
    
    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            summary_lines.append(line)
            continue
        if in_frontmatter:
            summary_lines.append(line)
            if line.strip() == "---":
                in_frontmatter = False
                frontmatter_closed = True
            continue
        
        # Capture title, status tags, and initial summary sections
        summary_lines.append(line)
        if len(summary_lines) >= max_lines:
            break
        # Stop at secondary major headers if not in frontmatter
        if frontmatter_closed and line.strip().startswith("## 2."):
            summary_lines.pop() # Remove the start of deep section
            break

    return "\n".join(summary_lines)


def extract_index(target: Path) -> str:
    """Extract table of contents or high-level mapping from target directory or index file."""
    root = get_project_root()
    target_path = (root / target).resolve() if not target.is_absolute() else target

    if target_path.is_file():
        return extract_summary(target_path, max_lines=40)

    # Check for canonical INDEX.md or README.md
    index_file = target_path / "INDEX.md"
    if not index_file.exists():
        index_file = target_path / "README.md"
    if index_file.exists():
        return extract_summary(index_file, max_lines=50)

    # Otherwise list markdown files and their top-level headers
    results = [f"Index of {target}:"]
    for md_file in sorted(target_path.glob("**/*.md")):
        if "ARCHIVE" in md_file.parts:
            continue
        try:
            rel = md_file.relative_to(root)
            first_header = "No header"
            for line in md_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    first_header = line.replace("# ", "").strip()
                    break
            results.append(f"- {rel}: {first_header}")
        except Exception:
            continue
    return "\n".join(results[:50])


def main():
    parser = argparse.ArgumentParser(description="Progressive Disclosure CLI")
    parser.add_argument("--level", choices=["index", "summary", "deep"], default="summary",
                        help="Granularity tier: index (Tier 1), summary (Tier 2), deep (Tier 3)")
    parser.add_argument("--file", type=str, help="Specific file to disclose")
    parser.add_argument("--target", type=str, default="wiki/", help="Target directory for index")
    parser.add_argument("--max-lines", type=int, default=30, help="Max lines for summary level")
    args = parser.parse_args()

    root = get_project_root()

    if args.level == "index":
        target = Path(args.target)
        print(extract_index(target))
    elif args.level == "summary":
        if not args.file:
            print("Error: --file argument is required for summary level.", file=sys.stderr)
            sys.exit(1)
        file_path = (root / args.file).resolve() if not Path(args.file).is_absolute() else Path(args.file)
        print(extract_summary(file_path, max_lines=args.max_lines))
    elif args.level == "deep":
        if not args.file:
            print("Error: --file argument is required for deep level.", file=sys.stderr)
            sys.exit(1)
        file_path = (root / args.file).resolve() if not Path(args.file).is_absolute() else Path(args.file)
        if not file_path.exists():
            print(f"Error: {file_path} does not exist.", file=sys.stderr)
            sys.exit(1)
        print(file_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
