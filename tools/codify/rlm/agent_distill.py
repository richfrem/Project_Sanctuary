#!/usr/bin/env python3
"""
agent_distill.py (CLI)
=====================================

Purpose:
    Agent-powered RLM cache distiller. Reads files and accepts pre-computed
    summaries (from the agent itself) instead of calling Ollama. Designed to
    be driven by an AI agent that provides summaries via a JSON input file.
    
    This tool complements the Ollama-based distiller.py by leveraging
    frontier-model intelligence for higher quality summaries.

Layer: Codify / RLM

Usage Examples:
    # Read a file and print its content for the agent to summarize
    python tools/codify/rlm/agent_distill.py read --file tools/cli.py
    
    # Apply agent-provided summaries from a JSON file
    python tools/codify/rlm/agent_distill.py apply --input /tmp/agent_summaries.json --cache-type tool
    python tools/codify/rlm/agent_distill.py apply --input /tmp/agent_summaries.json --cache-type sanctuary
    
    # List files needing distillation (failed or missing)
    python tools/codify/rlm/agent_distill.py list-needed --cache-type sanctuary
    python tools/codify/rlm/agent_distill.py list-needed --cache-type tool

CLI Arguments:
    read        : Read a file and output content for agent summarization
    apply       : Apply summaries from a JSON file to the cache
    list-needed : List files that need (re-)distillation

Script Dependencies:
    - tools/codify/rlm/rlm_config.py (Configuration)
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Setup project root
current_dir = Path(__file__).parent.resolve()
PROJECT_ROOT = current_dir.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.codify.rlm.rlm_config import (
    RLMConfig, collect_files, load_cache, save_cache, compute_hash
)

SUMMARY_CACHE_PATH = PROJECT_ROOT / ".agent" / "learning" / "rlm_summary_cache.json"
TOOL_CACHE_PATH = PROJECT_ROOT / ".agent" / "learning" / "rlm_tool_cache.json"


def cmd_read(args):
    """Read a file and output its content for agent summarization."""
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    
    try:
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_path = str(file_path)
    
    print(f"=== FILE: {rel_path} ===")
    print(f"=== SIZE: {len(content)} chars ===")
    print(content[:12000])  # Same truncation as distiller.py
    if len(content) > 12000:
        print("\n...[TRUNCATED]...")


def cmd_apply(args):
    """Apply agent-provided summaries to the cache."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    with open(input_path, "r") as f:
        summaries = json.load(f)
    
    cache_type = args.cache_type
    if cache_type == "tool":
        cache_path = TOOL_CACHE_PATH
    else:
        cache_path = SUMMARY_CACHE_PATH
    
    cache = load_cache(cache_path)
    applied = 0
    
    for file_path, summary in summaries.items():
        # Read the actual file to compute hash
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            content_hash = compute_hash(content)
            file_mtime = full_path.stat().st_mtime
        else:
            content_hash = f"agent_distilled_{datetime.now().strftime('%Y_%m_%d')}"
            file_mtime = None
        
        entry = {
            "hash": content_hash,
            "summary": summary,
            "summarized_at": datetime.now().isoformat()
        }
        if file_mtime:
            entry["file_mtime"] = file_mtime
            
        cache[file_path] = entry
        applied += 1
        print(f"  ✅ {file_path}")
    
    save_cache(cache, cache_path)
    print(f"\n{'='*50}")
    print(f"Applied {applied} summaries to {cache_path.name}")


def cmd_list_needed(args):
    """List files that need (re-)distillation."""
    cache_type = args.cache_type
    
    if cache_type == "tool":
        cache_path = TOOL_CACHE_PATH
        config = RLMConfig(run_type="tool")
        files = collect_files(config)
    else:
        cache_path = SUMMARY_CACHE_PATH
        # For sanctuary, manually collect from rlm_manifest.json
        manifest_path = PROJECT_ROOT / "tools" / "standalone" / "rlm-factory" / "rlm_manifest.json"
        manifest = json.load(open(manifest_path))
        files = []
        for td in manifest["target_directories"]:
            p = PROJECT_ROOT / td["path"]
            if not p.exists():
                continue
            for ft in td.get("file_types", [".md"]):
                for f in p.rglob(f"*{ft}"):
                    skip = False
                    for ex in manifest.get("exclude_patterns", []):
                        if ex.replace("**/", "") in str(f):
                            skip = True
                            break
                    if not skip and f.is_file():
                        files.append(f)
        # Add core files
        for cf in manifest.get("core_files", []):
            p = PROJECT_ROOT / cf["path"]
            if p.exists():
                files.append(p)
    
    cache = load_cache(cache_path)
    
    # Find missing or failed
    missing = []
    failed = []
    stale = []
    
    for f in files:
        try:
            rel = f.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(f)
        
        if rel not in cache:
            missing.append(rel)
        elif cache[rel].get("summary") == "[DISTILLATION FAILED]":
            failed.append(rel)
    
    # Deduplicate
    missing = sorted(set(missing))
    failed = sorted(set(failed))
    
    print(f"Cache: {cache_path.name} ({len(cache)} entries)")
    print(f"Scope: {len(files)} files")
    print(f"Missing: {len(missing)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\n--- FAILED ({len(failed)}) ---")
        for f in failed:
            print(f"  ❌ {f}")
    
    if missing:
        print(f"\n--- MISSING ({len(missing)}) ---")
        for f in missing:
            print(f"  ⚠️  {f}")
    
    if args.json_output:
        output = {"missing": missing, "failed": failed}
        print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Agent-powered RLM distiller")
    subparsers = parser.add_subparsers(dest="command")
    
    # read
    read_parser = subparsers.add_parser("read", help="Read a file for agent summarization")
    read_parser.add_argument("--file", required=True, help="File to read")
    
    # apply
    apply_parser = subparsers.add_parser("apply", help="Apply agent summaries to cache")
    apply_parser.add_argument("--input", required=True, help="JSON file with summaries")
    apply_parser.add_argument("--cache-type", choices=["tool", "sanctuary"], default="sanctuary")
    
    # list-needed
    list_parser = subparsers.add_parser("list-needed", help="List files needing distillation")
    list_parser.add_argument("--cache-type", choices=["tool", "sanctuary"], default="sanctuary")
    list_parser.add_argument("--json", dest="json_output", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "read":
        cmd_read(args)
    elif args.command == "apply":
        cmd_apply(args)
    elif args.command == "list-needed":
        cmd_list_needed(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
