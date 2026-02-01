#!/usr/bin/env python3
"""
fetch_tool_context.py (CLI)
=====================================

Purpose:
    Retrieves the "Gold Standard" tool definition from the RLM Tool Cache
    and formats it into an Agent-readable "Manual Page".
    
    This is the second step of the Late-Binding Protocol: after query_cache.py
    finds a tool, this script provides the detailed context needed to use it.

Layer: Retrieve

Usage Examples:
    python tools/retrieve/rlm/fetch_tool_context.py --file tools/cli.py
    python tools/retrieve/rlm/fetch_tool_context.py --file scripts/domain_cli.py

CLI Arguments:
    --file : Path to the tool script (required, e.g., tools/cli.py)

Output:
    Markdown-formatted technical specification to stdout:
    - Purpose
    - Usage
    - Arguments
    - Inputs/Outputs
    - Dependencies

Input Files:
    - .agent/learning/rlm_tool_cache.json (The Cache)

Key Functions:
    - fetch_context(): Loads entry from cache or falls back to docstring parsing.
    - format_context(): Formats JSON into Markdown manual.

Script Dependencies:
    - tools/codify/rlm/rlm_config.py: RLM configuration and cache loading

Consumed by:
    - Agent during Late-Binding tool discovery flow
"""
import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to sys.path to allow imports
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parents[2]
sys.path.append(str(project_root))

try:
    from tools.codify.rlm.rlm_config import RLMConfig, load_cache, PROJECT_ROOT
except ImportError:
    print("❌ Critical Error: Could not import RLM Configuration.")
    print(f"   Ensure 'tools/codify/rlm/rlm_config.py' exists relative to {current_dir}")
    sys.exit(1)

def safe_print(msg):
    """Print with fallback for Windows consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('utf-8', 'backslashreplace').decode())

def format_as_manual(file_path: str, data: dict):
    """Format the JSON tool definition into a Markdown Manual Page."""
    
    safe_print(f"\n# 🛠️ Tool Manual: {file_path}")
    safe_print(f"**Layer**: {data.get('layer', 'Unknown')}")
    safe_print(f"**Type**: {', '.join(data.get('supported_object_types', ['Generic']))}")
    safe_print("-" * 50)
    
    safe_print(f"\n## 🎯 Purpose")
    safe_print(data.get("purpose", "No purpose defined."))
    
    safe_print(f"\n## 💻 Usage")
    usage = data.get("usage")
    if isinstance(usage, list):
        for line in usage:
            safe_print(line)
    else:
        safe_print(usage or "No usage examples provided.")
        
    safe_print(f"\n## ⚙️ Arguments")
    args = data.get("args")
    if args and isinstance(args, list):
        for arg in args:
            if isinstance(arg, dict):
                safe_print(f"- **{arg.get('name')}**: {arg.get('description')}")
            else:
                safe_print(f"- `{arg}`")
    elif args:
         safe_print(str(args))
    else:
        safe_print("No arguments defined.")

    safe_print(f"\n## 📥 Inputs")
    inputs = data.get("inputs", [])
    for i in inputs:
        safe_print(f"- {i}")
        
    safe_print(f"\n## 📤 Outputs")
    outputs = data.get("outputs", [])
    for o in outputs:
        safe_print(f"- {o}")
        
    safe_print(f"\n## 🔗 Dependencies")
    deps = data.get("dependencies", [])
    for d in deps:
        safe_print(f"- {d}")
        
    safe_print("-" * 50)
    safe_print("USER INSTRUCTION: Inject this capability into your context for this turn.")

def main():
    parser = argparse.ArgumentParser(description="Fetch Tool Context from RLM Cache")
    parser.add_argument("--file", required=True, help="Path to the tool script (e.g., tools/cli.py)")
    args = parser.parse_args()
    
    # Initialize RLM in "Tool" mode
    try:
        config = RLMConfig(run_type="tool")
    except Exception as e:
        print(f"❌ Failed to initialize RLM Config: {e}")
        sys.exit(1)
        
    cache = load_cache(config.cache_path)
    
    # Normalize path
    target = args.file.replace("\\", "/") # Force POSIX
    
    # Try direct match
    entry = cache.get(target)
    
    if not entry:
        # Try resolving relative to root
        try:
             resolved = (Path.cwd() / args.file).resolve().relative_to(PROJECT_ROOT).as_posix()
             entry = cache.get(resolved)
        except Exception:
            pass
            
    if not entry:
        print(f"❌ Tool not found in cache: {target}")
        print(f"   Cache path: {config.cache_path}")
        print("   Run 'python tools/retrieve/rlm/query_cache.py --type tool \"term\"' to search.")
        sys.exit(1)
        
    # Parse the summary JSON
    try:
        tool_def = json.loads(entry["summary"])
        format_as_manual(target, tool_def)
    except json.JSONDecodeError:
        print(f"⚠️  Error: Tool summary is not valid JSON. Showing raw content:")
        print(entry["summary"])

if __name__ == "__main__":
    main()
