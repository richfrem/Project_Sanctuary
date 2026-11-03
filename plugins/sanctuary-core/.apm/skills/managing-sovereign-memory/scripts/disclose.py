#!/usr/bin/env python3
"""disclose.py - Progressive Disclosure CLI for Sovereign AI Memory.
Purpose: Retrieves knowledge at specific levels of granularity (Tier 1: Index, Tier 2: Summary, Tier 3: Deep) to preserve the agent's context window.
Layer: sanctuary-core / managing-sovereign-memory (Layer 2 Progressive Disclosure Retrieval).
Key Inputs: Markdown files in LEARNING/ and protocols/.
Key Outputs: Plain-text or structured JSON extraction with metadata, word counts, and byte offsets.
Usage Examples:
    python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 1
    python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 2 --path LEARNING/concepts/sovereign_memory_engine.md
    python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 3 --path protocols/128_Hardened_Learning_Loop_v3.0.md
    python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 2 --path LEARNING/concepts/sovereign_memory_engine.md --format json
Key Functions: get_project_root(), normalize_tier(), parse_frontmatter(), extract_summary(), extract_index(), main().
"""

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


def normalize_tier(tier_str: str | None, level_str: str | None) -> str:
    val = tier_str or level_str or "summary"
    val = str(val).lower().strip()
    mapping = {
        "1": "index",
        "tier 1": "index",
        "index": "index",
        "2": "summary",
        "tier 2": "summary",
        "summary": "summary",
        "3": "deep",
        "tier 3": "deep",
        "deep": "deep"
    }
    return mapping.get(val, "summary")


def parse_frontmatter(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separates YAML frontmatter from document body."""
    meta: dict[str, str] = {}
    if not lines or lines[0].strip() != "---":
        return meta, lines

    body_start = 1
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line == "---":
            body_start = i + 1
            break
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"').strip("'")

    return meta, lines[body_start:]


def extract_summary(file_path: Path, max_lines: int = 30, max_chars: int | None = None) -> dict:
    """Extract executive summary avoiding frontmatter and noisy diagram blocks."""
    if not file_path.exists() or not file_path.is_file():
        return {"error": f"File '{file_path}' not found", "status": 404}

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"Error reading file '{file_path}': {e}", "status": 500}

    lines = content.splitlines()
    meta, body_lines = parse_frontmatter(lines)

    summary_lines = []
    in_code_block = False

    for line in body_lines:
        stripped = line.strip()

        # Track fenced code blocks (Mermaid, bash, etc.)
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            # Don't let code/diagram blocks dominate summary
            if in_code_block:
                summary_lines.append(f"[Diagram / Code block: {stripped.replace('```', '').strip() or 'code'}]")
                continue
            else:
                continue

        if in_code_block:
            continue

        # Stop at secondary major headers (section 2 onwards)
        if stripped.startswith("## 2.") or stripped.startswith("## II.") or stripped.startswith("## Section 2"):
            break

        summary_lines.append(line)
        if len(summary_lines) >= max_lines:
            break

    # Strip empty lines from edges
    while summary_lines and not summary_lines[0].strip():
        summary_lines.pop(0)
    while summary_lines and not summary_lines[-1].strip():
        summary_lines.pop()

    text_output = "\n".join(summary_lines)
    if max_chars and len(text_output) > max_chars:
        text_output = text_output[:max_chars] + "\n... [truncated by max_chars]"

    return {
        "path": str(file_path),
        "status": "ok",
        "metadata": meta,
        "summary": text_output,
        "lines_returned": len(summary_lines)
    }


def extract_index(target_path: Path, max_chars: int | None = None) -> dict:
    """Extract table of contents or high-level mapping from target directory or index file."""
    root = get_project_root()
    resolved = (root / target_path).resolve() if not target_path.is_absolute() else target_path

    if resolved.is_file():
        summary_res = extract_summary(resolved, max_lines=40, max_chars=max_chars)
        return {
            "type": "index_file",
            "path": str(resolved.relative_to(root) if resolved.is_relative_to(root) else resolved),
            "content": summary_res.get("summary", ""),
            "metadata": summary_res.get("metadata", {})
        }

    # Check for canonical INDEX.md or README.md in directory
    index_file = resolved / "INDEX.md"
    if not index_file.exists():
        index_file = resolved / "README.md"
    if index_file.exists():
        summary_res = extract_summary(index_file, max_lines=40, max_chars=max_chars)
        return {
            "type": "index_file",
            "path": str(index_file.relative_to(root) if index_file.is_relative_to(root) else index_file),
            "content": summary_res.get("summary", ""),
            "metadata": summary_res.get("metadata", {})
        }

    # Otherwise discover markdown files and their primary titles
    entries = []
    if resolved.exists():
        for md_file in sorted(resolved.glob("**/*.md")):
            if "ARCHIVE" in md_file.parts or ".agent" in md_file.parts:
                continue
            try:
                rel = str(md_file.relative_to(root))
                first_header = "No header"
                for line in md_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("# "):
                        first_header = line.replace("# ", "").strip()
                        break
                entries.append({"path": rel, "title": first_header})
            except Exception:
                continue

    text_lines = [f"=== Sovereign Knowledge Index: {target_path} ==="]
    for e in entries[:60]:
        text_lines.append(f"- {e['path']}: {e['title']}")

    text_output = "\n".join(text_lines)
    if max_chars and len(text_output) > max_chars:
        text_output = text_output[:max_chars] + "\n... [truncated by max_chars]"

    return {
        "type": "directory_index",
        "target": str(target_path),
        "total_entries": len(entries),
        "content": text_output,
        "entries": entries[:60]
    }


def main():
    parser = argparse.ArgumentParser(description="Progressive Disclosure CLI for Sovereign AI Memory")
    parser.add_argument("--tier", "-t", type=str, help="Granularity tier: 1 (index), 2 (summary), 3 (deep)")
    parser.add_argument("--level", "-l", type=str, help="Alias for --tier: index, summary, deep")
    parser.add_argument("--path", "-p", type=str, help="Specific file or directory path")
    parser.add_argument("--file", "-f", type=str, help="Alias for --path")
    parser.add_argument("--target", type=str, default="LEARNING/concepts/", help="Target directory for index (Tier 1)")
    parser.add_argument("--max-lines", type=int, default=30, help="Max lines for summary extraction")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum characters to return")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    args = parser.parse_args()

    tier = normalize_tier(args.tier, args.level)
    target_path_str = args.path or args.file
    root = get_project_root()

    if tier == "index":
        target = Path(target_path_str or args.target)
        result = extract_index(target, max_chars=args.max_chars)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(result["content"])
        sys.exit(0)

    elif tier == "summary":
        if not target_path_str:
            err_msg = "Error: --path or --file argument is required for Tier 2 summary."
            if args.format == "json":
                print(json.dumps({"error": err_msg, "status": 400}))
            else:
                print(err_msg, file=sys.stderr)
            sys.exit(1)

        file_path = (root / target_path_str).resolve() if not Path(target_path_str).is_absolute() else Path(target_path_str)
        result = extract_summary(file_path, max_lines=args.max_lines, max_chars=args.max_chars)

        if "error" in result:
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(result["summary"])
        sys.exit(0)

    elif tier == "deep":
        if not target_path_str:
            err_msg = "Error: --path or --file argument is required for Tier 3 deep disclosure."
            if args.format == "json":
                print(json.dumps({"error": err_msg, "status": 400}))
            else:
                print(err_msg, file=sys.stderr)
            sys.exit(1)

        file_path = (root / target_path_str).resolve() if not Path(target_path_str).is_absolute() else Path(target_path_str)
        if not file_path.exists() or not file_path.is_file():
            err_msg = f"File '{file_path}' not found."
            if args.format == "json":
                print(json.dumps({"error": err_msg, "status": 404}))
            else:
                print(f"Error: {err_msg}", file=sys.stderr)
            sys.exit(1)

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            err_msg = f"Error reading file '{file_path}': {e}"
            if args.format == "json":
                print(json.dumps({"error": err_msg, "status": 500}))
            else:
                print(f"Error: {err_msg}", file=sys.stderr)
            sys.exit(1)

        if args.max_chars and len(content) > args.max_chars:
            content = content[:args.max_chars] + "\n... [truncated by max_chars]"

        if args.format == "json":
            print(json.dumps({
                "path": str(target_path_str),
                "status": "ok",
                "length_chars": len(content),
                "content": content
            }, indent=2))
        else:
            print(content)
        sys.exit(0)


if __name__ == "__main__":
    main()
