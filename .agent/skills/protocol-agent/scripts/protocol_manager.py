#!/usr/bin/env python3
"""
protocol_manager.py — Protocol Document Manager
=================================================

Purpose:
    Create, list, search, update, and view Protocol documents.
    Consolidates logic from mcp_servers/protocol/ into a standalone CLI.

Layer: Plugin / Protocol-Manager

Usage:
    python3 protocol_manager.py create "Title" --content "..." --status PROPOSED
    python3 protocol_manager.py list [--limit N] [--status STATUS]
    python3 protocol_manager.py get N
    python3 protocol_manager.py search "query"
    python3 protocol_manager.py update N --status CANONICAL --reason "Approved"
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
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
PROTOCOL_DIR = PROJECT_ROOT / "01_PROTOCOLS"

# --- Models ---

class ProtocolStatus(str, Enum):
    PROPOSED = "PROPOSED"
    CANONICAL = "CANONICAL"
    DEPRECATED = "DEPRECATED"

PROTOCOL_TEMPLATE = """# Protocol {number}: {title}

**Status:** {status}
**Classification:** {classification}
**Version:** {version}
**Authority:** {authority}
{linked_protocols_line}
---

{content}
"""

# --- Validator ---

def _get_next_number(base_dir: Path) -> int:
    """Get next available protocol number."""
    if not base_dir.exists():
        return 1
    max_num = 0
    for f in base_dir.iterdir():
        match = re.match(r"(\d+)_", f.name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1


def _find_protocol_file(base_dir: Path, number: int) -> Optional[Path]:
    """Find file path for a protocol number."""
    if not base_dir.exists():
        return None
    for f in base_dir.iterdir():
        match = re.match(r"(\d+)_", f.name)
        if match and int(match.group(1)) == number:
            return f
    return None


def _parse_protocol(content: str, number: int) -> Dict[str, Any]:
    """Parse markdown content into protocol dict."""
    lines = content.split("\n")
    metadata: Dict[str, str] = {}
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("**Status:**"):
            metadata["status"] = line.replace("**Status:**", "").strip()
        elif line.startswith("**Classification:**"):
            metadata["classification"] = line.replace("**Classification:**", "").strip()
        elif line.startswith("**Version:**"):
            metadata["version"] = line.replace("**Version:**", "").strip()
        elif line.startswith("**Authority:**"):
            metadata["authority"] = line.replace("**Authority:**", "").strip()
        elif line.startswith("**Linked Protocols:**"):
            metadata["linked_protocols"] = line.replace("**Linked Protocols:**", "").strip()
        elif line.strip() == "---":
            body_start = i + 1
            break

    title = "Unknown Protocol"
    for line in lines:
        if line.startswith("# Protocol"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                title = parts[1].strip()
            break

    return {
        "number": number,
        "title": title,
        "status": metadata.get("status", "PROPOSED"),
        "classification": metadata.get("classification", ""),
        "version": metadata.get("version", "1.0"),
        "authority": metadata.get("authority", ""),
        "linked_protocols": metadata.get("linked_protocols", ""),
        "content": "\n".join(lines[body_start:]).strip() if body_start > 0 else content
    }


# --- Operations ---

def create_protocol(title: str, content: str, status: str = "PROPOSED",
                    classification: str = "Internal", version: str = "1.0",
                    authority: str = "Project Sanctuary", linked: str = ""):
    """Create a new protocol document."""
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

    number = _get_next_number(PROTOCOL_DIR)
    slug = title.replace(" ", "_").replace("-", "_")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    filename = f"{number:02d}_{slug}.md"
    filepath = PROTOCOL_DIR / filename

    linked_line = f"**Linked Protocols:** {linked}" if linked else ""
    file_content = PROTOCOL_TEMPLATE.format(
        number=number, title=title, status=status,
        classification=classification, version=version,
        authority=authority, linked_protocols_line=linked_line,
        content=content
    )

    filepath.write_text(file_content, encoding='utf-8')
    print(f"✅ Created Protocol {number}: {title}")
    print(f"   Path: {filepath}")
    return filepath


def list_protocols(limit: int = None, status: str = None):
    """List protocols."""
    if not PROTOCOL_DIR.exists():
        print("📂 No protocols directory found.")
        return

    protocols = []
    for f in sorted(PROTOCOL_DIR.iterdir()):
        if not f.name.endswith(".md") or f.name.startswith("."):
            continue
        match = re.match(r"(\d+)_", f.name)
        if match:
            number = int(match.group(1))
            try:
                p = _parse_protocol(f.read_text(encoding='utf-8'), number)
                if status is None or p["status"] == status:
                    protocols.append(p)
            except Exception:
                continue

    if limit:
        protocols = protocols[-limit:]

    if not protocols:
        print("📂 No protocols found.")
        return

    print(f"\n📋 Protocols ({len(protocols)}):\n")
    for p in protocols:
        status_icon = {"PROPOSED": "🟡", "CANONICAL": "🟢", "DEPRECATED": "🔴"}.get(p["status"], "⚪")
        print(f"  {status_icon} {p['number']:03d}  {p['title'][:50]:50}  [{p['status']}]")


def get_protocol(number: int):
    """View a specific protocol."""
    filepath = _find_protocol_file(PROTOCOL_DIR, number)
    if not filepath:
        print(f"❌ Protocol {number} not found.")
        return
    print(filepath.read_text(encoding='utf-8'))


def search_protocols(query: str):
    """Search protocols by keyword."""
    if not PROTOCOL_DIR.exists():
        print("📂 No protocols directory.")
        return

    results = []
    for f in sorted(PROTOCOL_DIR.iterdir()):
        if not f.name.endswith(".md") or f.name.startswith("."):
            continue
        try:
            content = f.read_text(encoding='utf-8')
            if query.lower() in content.lower():
                match = re.match(r"(\d+)_", f.name)
                if match:
                    p = _parse_protocol(content, int(match.group(1)))
                    results.append(p)
        except Exception:
            continue

    if not results:
        print(f"❌ No protocols matching '{query}'")
    else:
        print(f"\n🔍 {len(results)} protocol(s) matching '{query}':\n")
        for p in results:
            print(f"  {p['number']:03d}  {p['title'][:50]:50}  [{p['status']}]")


def update_protocol(number: int, status: str = None, reason: str = ""):
    """Update a protocol's status or fields."""
    filepath = _find_protocol_file(PROTOCOL_DIR, number)
    if not filepath:
        print(f"❌ Protocol {number} not found.")
        return

    content = filepath.read_text(encoding='utf-8')
    p = _parse_protocol(content, number)

    if status:
        p["status"] = status

    linked_line = f"**Linked Protocols:** {p['linked_protocols']}" if p.get("linked_protocols") else ""
    new_content = PROTOCOL_TEMPLATE.format(
        number=p["number"], title=p["title"], status=p["status"],
        classification=p["classification"], version=p["version"],
        authority=p["authority"], linked_protocols_line=linked_line,
        content=p["content"]
    )

    filepath.write_text(new_content, encoding='utf-8')
    print(f"✅ Updated Protocol {number}")
    if status:
        print(f"   Status → {status}")
    if reason:
        print(f"   Reason: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Protocol Document Manager")
    subparsers = parser.add_subparsers(dest="command")

    create_p = subparsers.add_parser("create", help="Create new protocol")
    create_p.add_argument("title", help="Protocol title")
    create_p.add_argument("--content", required=True, help="Protocol content")
    create_p.add_argument("--status", default="PROPOSED", help="Status")
    create_p.add_argument("--classification", default="Internal", help="Classification")
    create_p.add_argument("--version", default="1.0", help="Version")
    create_p.add_argument("--authority", default="Project Sanctuary", help="Authority")
    create_p.add_argument("--linked", default="", help="Linked protocol numbers")

    list_p = subparsers.add_parser("list", help="List protocols")
    list_p.add_argument("--limit", type=int, help="Show last N")
    list_p.add_argument("--status", help="Filter by status")

    get_p = subparsers.add_parser("get", help="View protocol")
    get_p.add_argument("number", type=int, help="Protocol number")

    search_p = subparsers.add_parser("search", help="Search protocols")
    search_p.add_argument("query", help="Search query")

    update_p = subparsers.add_parser("update", help="Update protocol")
    update_p.add_argument("number", type=int, help="Protocol number")
    update_p.add_argument("--status", help="New status")
    update_p.add_argument("--reason", default="", help="Update reason")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "create":
        create_protocol(args.title, args.content, args.status, args.classification,
                       args.version, args.authority, args.linked)
    elif args.command == "list":
        list_protocols(args.limit, args.status)
    elif args.command == "get":
        get_protocol(args.number)
    elif args.command == "search":
        search_protocols(args.query)
    elif args.command == "update":
        update_protocol(args.number, args.status, args.reason)


if __name__ == "__main__":
    main()
