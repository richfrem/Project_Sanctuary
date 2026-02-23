#!/usr/bin/env python3
"""
<<<<<<< HEAD
query_cache.py
=====================================

Purpose:
    RLM Search: Instant keyword-based lookup against the semantic ledger.
    Searches entry paths and summary text to surface relevant cached records.

Layer: Curate / Rlm

Usage:
    python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --profile plugins "rlm"
    python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --profile tools --list
    python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --profile plugins "factory" --json

Related:
    - rlm_config.py (configuration & cache utilities)
    - distiller.py (cache population)
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# ============================================================
# PATHS
# File is at: plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py
# Root is 6 levels up (scripts→rlm-curator→skills→rlm-factory→plugins→ROOT)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from rlm_config import RLMConfig, load_cache
except ImportError as e:
    print(f"❌ Could not import RLMConfig from {SCRIPT_DIR}: {e}")
    sys.exit(1)


# ----------------------------------------------------------
# search_cache — keyword search across paths and summaries
# ----------------------------------------------------------
def search_cache(
    term: str,
    config: RLMConfig,
    show_summary: bool = True,
    output_json: bool = False
) -> None:
    """
    Search the RLM cache for entries matching the given term.

    Matches against both the file path and the summary text. Results are
    sorted by path and printed to stdout (or serialized as JSON).

    Args:
        term: Keyword to search for (case-insensitive).
        config: Active RLMConfig providing the cache path.
        show_summary: If True, print a preview of the summary text.
        output_json: If True, serialize results as JSON instead of formatted text.
    """
    data = load_cache(config.cache_path)
    term_lower = term.lower()

    if not output_json:
        print(f"🔍 Searching [{config.profile_name.upper()}] for: '{term}'...")

    matches: List[Dict[str, Any]] = []
    for rel_path, entry in data.items():
        if term_lower in rel_path.lower() or term_lower in entry.get("summary", "").lower():
            matches.append({"path": rel_path, "entry": entry})

    matches.sort(key=lambda x: x["path"])
=======
query_cache.py (CLI)
=====================================

Purpose:
    RLM Search: Instant O(1) semantic search of the ledger.

Layer: Curate / Rlm

Usage Examples:
    python plugins/rlm-factory/scripts/query_cache.py --help
    python plugins/rlm-factory/scripts/query_cache.py "Project Sanctuary"

Supported Object Types:
    - Generic

CLI Arguments:
    term            : Search term (ID, filename, or content keyword)
    --list          : List all cached files
    --no-summary    : Hide summary text
    --json          : Output results as JSON

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - load_cache(): No description.
    - search_cache(): No description.
    - list_cache(): No description.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import json
import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path to find tools package
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from rlm_config import RLMConfig
except ImportError:
    try:
        from tools.tool_inventory.rlm_config import RLMConfig
    except ImportError:
        print("❌ Could not import RLMConfig (tried local and tools.tool_inventory)")
        sys.exit(1)

def load_cache(config: RLMConfig):
    if not config.cache_path.exists():
        print(f"❌ Cache file not found: {config.cache_path}")
        return {}
    try:
        with open(config.cache_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error decoding cache JSON")
        return {}

def search_cache(term, config: RLMConfig, show_summary=True, return_data=False, output_json=False):
    data = load_cache(config)
    matches = []
    
    if not output_json and not return_data:
        print(f"🔍 Searching RLM Cache [{config.type.upper()}] for: '{term}'...")
    
    for relative_path, entry in data.items():
        # Match against file path
        if term.lower() in relative_path.lower():
            matches.append({"path": relative_path, "entry": entry})
            continue
            
        # Match against summary content
        if term.lower() in entry.get('summary', '').lower():
            matches.append({"path": relative_path, "entry": entry})
            continue
            
        # Match against ID or Hash (less likely but useful)
        if term.lower() in entry.get('content_hash', '').lower():
            matches.append({"path": relative_path, "entry": entry})
            
    # Sort matches by path for consistency
    matches.sort(key=lambda x: x['path'])

    if return_data:
        return matches
>>>>>>> origin/main

    if output_json:
        print(json.dumps(matches, indent=2))
        return

    if not matches:
<<<<<<< HEAD
        print("   No matches found.")
        return

    print(f"✅ Found {len(matches)} match(es):\n")
    for match in matches:
        path = match["path"]
        entry = match["entry"]
        print(f"📄 {path}")
        print(f"   🕒 Indexed: {entry.get('summarized_at', 'Unknown')}")
        if show_summary:
            summary = entry.get("summary", "No summary.")
            preview = (summary[:300] + "...") if len(summary) > 300 else summary
            print(f"   📝 {preview}")
        print("-" * 50)


# ----------------------------------------------------------
# list_cache — enumerate all entries in a cache
# ----------------------------------------------------------
def list_cache(config: RLMConfig) -> None:
    """
    List all file paths currently indexed in the cache.

    Args:
        config: Active RLMConfig providing the cache path.
    """
    data = load_cache(config.cache_path)
    print(f"📚 [{config.profile_name.upper()}] — {len(data)} entries:\n")
    for rel_path in sorted(data.keys()):
        print(f"   - {rel_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main() -> None:
    """Parse CLI arguments and dispatch to search_cache() or list_cache()."""
    parser = argparse.ArgumentParser(description="RLM Cache — keyword search and listing")
    parser.add_argument("term", nargs="?", help="Search term (filename fragment or content keyword)")
    parser.add_argument("--profile", required=True, help="RLM profile name (from rlm_profiles.json)")
    parser.add_argument("--list", action="store_true", help="List all cached file paths")
    parser.add_argument("--no-summary", action="store_true", help="Hide summary text in results")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()
    config = RLMConfig(profile_name=args.profile)

=======
        print("No matches found.")
        return

    print(f"✅ Found {len(matches)} matches:\n")
    for match in matches:
        path = match['path']
        entry = match['entry']
        timestamp = entry.get('summarized_at', 'Unknown Time')
        print(f"📄 {path}")
        print(f"   🕒 Last Indexed: {timestamp}")
        if show_summary:
            summary = entry.get('summary', 'No summary available.')
            if isinstance(summary, str):
                 print(f"   📝 Summary: {summary[:200]}..." if len(summary) > 200 else f"   📝 Summary: {summary}")
            else:
                 print(f"   📝 Summary: {json.dumps(summary, indent=2)}")
        print("-" * 50)

def list_cache(config: RLMConfig):
    data = load_cache(config)
    print(f"📚 RLM Cache [{config.type.upper()}] contains {len(data)} entries:\n")
    for relative_path in sorted(data.keys()):
        print(f"- {relative_path}")

def main():
    parser = argparse.ArgumentParser(description="Query RLM Cache")
    parser.add_argument("term", nargs="?", help="Search term (ID, filename, or content keyword)")
    # parser.add_argument("--type", choices=["project", "tool"], default="project", help="RLM Type (loads manifest from factory)")
    parser.add_argument("--list", action="store_true", help="List all cached files")
    parser.add_argument("--no-summary", action="store_true", help="Hide summary text")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Load Config based on Type
    config = RLMConfig(run_type="project")
    
>>>>>>> origin/main
    if args.list:
        list_cache(config)
    elif args.term:
        search_cache(args.term, config, show_summary=not args.no_summary, output_json=args.json)
    else:
        parser.print_help()

<<<<<<< HEAD

=======
>>>>>>> origin/main
if __name__ == "__main__":
    main()
