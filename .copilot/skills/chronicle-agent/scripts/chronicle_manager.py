#!/usr/bin/env python3
"""
chronicle_manager.py — Living Chronicle Manager
=================================================

Purpose:
    Create, list, search, and view Chronicle entries (project journal).
    Consolidates logic from mcp_servers/chronicle/ into a standalone CLI.

Layer: Plugin / Chronicle-Manager

Usage:
    python3 chronicle_manager.py create "Title" --content "..." [--author "Name"]
    python3 chronicle_manager.py list [--limit N]
    python3 chronicle_manager.py get N
    python3 chronicle_manager.py search "query"
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import date
from typing import List, Optional, Dict, Any
from enum import Enum

SCRIPT_DIR = Path(__file__).parent.resolve()
PLUGIN_ROOT = SCRIPT_DIR.parent.resolve()


def _find_project_root() -> Path:
    """Walk up to find the project root."""
    p = PLUGIN_ROOT
    for _ in range(10):
        if (p / ".git").exists() or (p / ".agent").exists():
            return p
        p = p.parent
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
CHRONICLE_DIR = PROJECT_ROOT / "02_LIVING_CHRONICLE"

# --- Models ---

class ChronicleStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    CANONICAL = "canonical"
    DEPRECATED = "deprecated"

class ChronicleClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"

CHRONICLE_TEMPLATE = """# Living Chronicle - Entry {number}

**Title:** {title}
**Date:** {date}
**Author:** {author}
**Status:** {status}
**Classification:** {classification}

---

{content}
"""

# --- Helpers ---

def _get_next_number(base_dir: Path) -> int:
    """Get next available entry number."""
    if not base_dir.exists():
        return 1
    max_num = 0
    for f in base_dir.iterdir():
        match = re.match(r"(\d{3})_", f.name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1


def _find_entry_file(base_dir: Path, number: int) -> Optional[Path]:
    """Find file path for an entry number."""
    if not base_dir.exists():
        return None
    for f in base_dir.iterdir():
        if f.name.startswith(f"{number:03d}_"):
            return f
    return None


def _parse_entry(content: str, number: int) -> Dict[str, Any]:
    """Parse markdown content into entry dict."""
    lines = content.split("\n")
    metadata: Dict[str, str] = {}
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("**Title:**"):
            metadata["title"] = line.replace("**Title:**", "").strip()
        elif line.startswith("**Date:**"):
            metadata["date"] = line.replace("**Date:**", "").strip()
        elif line.startswith("**Author:**"):
            metadata["author"] = line.replace("**Author:**", "").strip()
        elif line.startswith("**Status:**"):
            metadata["status"] = line.replace("**Status:**", "").strip()
        elif line.startswith("**Classification:**"):
            metadata["classification"] = line.replace("**Classification:**", "").strip()
        elif line.strip() == "---":
            body_start = i + 1
            break

    # Fallback title from H1 or H3
    if "title" not in metadata:
        for line in lines:
            if line.startswith("# "):
                metadata["title"] = line.lstrip("# ").strip()
                break
            elif line.startswith("### **Entry"):
                parts = line.split(":")
                if len(parts) > 1:
                    metadata["title"] = parts[1].replace("**", "").strip()
                break

    return {
        "number": number,
        "title": metadata.get("title", "Unknown Title"),
        "date": metadata.get("date", ""),
        "author": metadata.get("author", ""),
        "status": metadata.get("status", "draft"),
        "classification": metadata.get("classification", "internal"),
        "content": "\n".join(lines[body_start:]).strip() if body_start > 0 else content
    }


# --- Operations ---

def create_entry(title: str, content: str, author: str = "Guardian",
                 status: str = "draft", classification: str = "internal"):
    """Create a new chronicle entry."""
    CHRONICLE_DIR.mkdir(parents=True, exist_ok=True)

    number = _get_next_number(CHRONICLE_DIR)
    slug = title.lower().replace(" ", "_").replace("-", "_")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    filename = f"{number:03d}_{slug}.md"
    filepath = CHRONICLE_DIR / filename

    today = date.today().isoformat()
    file_content = CHRONICLE_TEMPLATE.format(
        number=number, title=title, date=today,
        author=author, status=status,
        classification=classification, content=content
    )

    filepath.write_text(file_content, encoding='utf-8')
    print(f"✅ Created Chronicle Entry {number:03d}: {title}")
    print(f"   Path: {filepath}")


def list_entries(limit: int = 10):
    """List recent chronicle entries."""
    if not CHRONICLE_DIR.exists():
        print("📂 No chronicle directory found.")
        return

    entries = []
    for f in sorted(CHRONICLE_DIR.iterdir(), reverse=True):
        if not f.name.endswith(".md") or f.name.startswith("."):
            continue
        match = re.match(r"(\d{3})_", f.name)
        if match:
            number = int(match.group(1))
            try:
                e = _parse_entry(f.read_text(encoding='utf-8'), number)
                entries.append(e)
            except Exception:
                continue
        if len(entries) >= limit:
            break

    if not entries:
        print("📂 No chronicle entries found.")
        return

    print(f"\n📜 Chronicle Entries (showing {len(entries)}):\n")
    for e in entries:
        status_icon = {"draft": "📝", "published": "📗", "canonical": "🏛️", "deprecated": "🔴"}.get(e["status"], "⚪")
        print(f"  {status_icon} {e['number']:03d}  {e['date']:12}  {e['title'][:45]:45}  [{e['status']}]")


def get_entry(number: int):
    """View a specific chronicle entry."""
    filepath = _find_entry_file(CHRONICLE_DIR, number)
    if not filepath:
        print(f"❌ Chronicle entry {number} not found.")
        return
    print(filepath.read_text(encoding='utf-8'))


def search_entries(query: str):
    """Search chronicle entries by keyword."""
    if not CHRONICLE_DIR.exists():
        print("📂 No chronicle directory.")
        return

    results = []
    for f in sorted(CHRONICLE_DIR.iterdir()):
        if not f.name.endswith(".md") or f.name.startswith("."):
            continue
        try:
            content = f.read_text(encoding='utf-8')
            if query.lower() in content.lower():
                match = re.match(r"(\d{3})_", f.name)
                if match:
                    e = _parse_entry(content, int(match.group(1)))
                    results.append(e)
        except Exception:
            continue

    if not results:
        print(f"❌ No entries matching '{query}'")
    else:
        print(f"\n🔍 {len(results)} entry/entries matching '{query}':\n")
        for e in results:
            print(f"  {e['number']:03d}  {e['date']:12}  {e['title'][:45]:45}  [{e['status']}]")


def main():
    parser = argparse.ArgumentParser(description="Living Chronicle Manager")
    subparsers = parser.add_subparsers(dest="command")

    create_p = subparsers.add_parser("create", help="Create new entry")
    create_p.add_argument("title", help="Entry title")
    create_p.add_argument("--content", required=True, help="Entry content")
    create_p.add_argument("--author", default="Guardian", help="Author name")
    create_p.add_argument("--status", default="draft", help="Status")
    create_p.add_argument("--classification", default="internal", help="Classification")

    list_p = subparsers.add_parser("list", help="List recent entries")
    list_p.add_argument("--limit", type=int, default=10, help="Show last N")

    get_p = subparsers.add_parser("get", help="View entry")
    get_p.add_argument("number", type=int, help="Entry number")

    search_p = subparsers.add_parser("search", help="Search entries")
    search_p.add_argument("query", help="Search query")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "create":
        create_entry(args.title, args.content, args.author, args.status, args.classification)
    elif args.command == "list":
        list_entries(args.limit)
    elif args.command == "get":
        get_entry(args.number)
    elif args.command == "search":
        search_entries(args.query)


if __name__ == "__main__":
    main()
