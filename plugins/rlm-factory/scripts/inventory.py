#!/usr/bin/env python3
"""
inventory.py (CLI)
=====================================

Purpose:
    RLM Auditor: Reports coverage of the semantic ledger against the filesystem.
    Uses the Shared RLMConfig to dynamically switch between 'Legacy' (Documentation) and 'Tool' (CLI) audit modes.

Layer: Curate / Rlm

Supported Object Types:
    - RLM Cache (Legacy)
    - RLM Cache (Tool)

CLI Arguments:
    --type  : [legacy|tool] Selects the configuration profile (default: legacy).

Input Files:
    - .agent/learning/rlm_summary_cache.json (Legacy)
    - .agent/learning/rlm_tool_cache.json (Tool)
    - Filesystem targets (defined in manifests)
    - tool_inventory.json

Output:
    - Console report (Statistics, Missing Files, Stale Entries)

Key Functions:
    - audit_inventory(): Logic to compare cache keys against collected file paths.

Script Dependencies:
    - tools/codify/rlm/rlm_config.py
"""
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path to find tools package
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from tools.codify.rlm.rlm_config import RLMConfig, load_cache, collect_files
except ImportError:
    print("❌ Could not import RLMConfig from tools.codify.rlm.rlm_config")
    sys.exit(1)

def audit_inventory(config: RLMConfig):
    """Compare RLM cache against actual file system."""
    
    print(f"📊 Auditing RLM Inventory [{config.type.upper()}]...")
    print(f"   Cache: {config.cache_path.name}")
    
    # 1. Load Cache
    cache = load_cache(config.cache_path)
    cached_paths = set(cache.keys())
    
    # 2. Scan File System / Inventory
    fs_files = collect_files(config)
    
    # Convert absolute paths to relative keys matching cache format
    fs_paths = set()
    for f in fs_files:
        try:
            rel = str(f.relative_to(PROJECT_ROOT))
            fs_paths.add(rel)
        except ValueError:
            pass
            
    # 3. Compare
    missing_in_cache = fs_paths - cached_paths
    stale_in_cache = cached_paths - fs_paths
    
    # 4. Report
    print(f"\n📈 Statistics:")
    print(f"   Files on Disk/Inventory: {len(fs_paths)}")
    print(f"   Entries in Cache:        {len(cached_paths)}")
    percentage = (len(fs_paths & cached_paths)/len(fs_paths)*100) if fs_paths else 0
    print(f"   Coverage:                {len(fs_paths & cached_paths)} / {len(fs_paths)} ({percentage:.1f}%)")
    
    if missing_in_cache:
        print(f"\n❌ Missing from Cache ({len(missing_in_cache)}):")
        for p in sorted(list(missing_in_cache))[:10]:
             print(f"   - {p}")
        if len(missing_in_cache) > 10:
            print(f"   ... and {len(missing_in_cache) - 10} more.")
            
    if stale_in_cache:
        print(f"\n⚠️  Stale in Cache ({len(stale_in_cache)}):")
        for p in sorted(list(stale_in_cache))[:10]:
             print(f"   - {p}")
        if len(stale_in_cache) > 10:
             print(f"   ... and {len(stale_in_cache) - 10} more.")
             
    if not missing_in_cache and not stale_in_cache:
        print("\n✅ RLM Inventory is perfectly synchronized.")

def main():
    parser = argparse.ArgumentParser(description="Audit RLM Cache Coverage")
    parser.add_argument("--type", choices=["legacy", "tool"], default="legacy", help="RLM Type (loads manifest from factory)")
    
    args = parser.parse_args()
    
    # Load Config based on Type
    try:
        config = RLMConfig(run_type=args.type)
        audit_inventory(config)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()