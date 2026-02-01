#!/usr/bin/env python3
"""
query_cache.py (CLI)
=====================================

Purpose:
    RLM Search: Instant O(1) semantic search of the ledger.

Layer: Curate / Rlm

Usage Examples:
    python tools/retrieve/rlm/query_cache.py --help

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
import signal
import os
from pathlib import Path

# Fix BrokenPipeError when piping to head
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Add project root to sys.path to find tools package
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from tools.codify.rlm.rlm_config import RLMConfig
except ImportError:
    print("❌ Could not import RLMConfig from tools.codify.rlm.rlm_config")
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
        # Match against summary content
        summary_val = entry.get('summary', '')
        if isinstance(summary_val, str):
            if term.lower() in summary_val.lower():
                matches.append({"path": relative_path, "entry": entry})
                continue
        else:
            # If summary is complex (dict/list), search in its string representation
            if term.lower() in str(summary_val).lower():
                matches.append({"path": relative_path, "entry": entry})
                continue
            
        # Match against ID or Hash (less likely but useful)
        if term.lower() in entry.get('content_hash', '').lower():
            matches.append({"path": relative_path, "entry": entry})
            
    # Sort matches by path for consistency
    matches.sort(key=lambda x: x['path'])

    if return_data:
        return matches

    if output_json:
        print(json.dumps(matches, indent=2))
        return

    if not matches:
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
    parser.add_argument("--type", default="tool", help="RLM Type (loads manifest from factory)")
    parser.add_argument("--list", action="store_true", help="List all cached files")
    parser.add_argument("--no-summary", action="store_true", help="Hide summary text")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Load Config based on Type
    config = RLMConfig(run_type=args.type)
    
    if args.list:
        list_cache(config)
    elif args.term:
        search_cache(args.term, config, show_summary=not args.no_summary, output_json=args.json)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
