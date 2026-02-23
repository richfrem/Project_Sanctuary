#!/usr/bin/env python3
"""
<<<<<<< HEAD
inventory.py
=====================================

Purpose:
    RLM Auditor: Compares the semantic cache against the live filesystem to
    report coverage gaps. Identifies files that are uncached (missing) and
    cache entries whose source files have been deleted (stale).

Layer: Curate / Rlm

Usage:
    python plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py --profile plugins
    python plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py --profile tools

Related:
    - rlm_config.py (configuration & file collection)
    - distiller.py (cache population)
    - cleanup_cache.py (stale entry removal)
"""
import sys
import argparse
from pathlib import Path
from typing import Set

# ============================================================
# PATHS
# File is at: plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py
# Root is 6 levels up (scripts→rlm-curator→skills→rlm-factory→plugins→ROOT)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from rlm_config import RLMConfig, load_cache, collect_files
except ImportError as e:
    print(f"❌ Could not import RLMConfig from {SCRIPT_DIR}: {e}")
    sys.exit(1)


# ----------------------------------------------------------
# audit_inventory — coverage report for a single profile
# ----------------------------------------------------------
def audit_inventory(config: RLMConfig) -> None:
    """
    Compare the RLM cache against the live filesystem and print a coverage report.

    Identifies:
      - Files on disk that are not yet in the cache (missing).
      - Cache entries whose source files no longer exist (stale).

    Args:
        config: Active RLMConfig defining the cache and manifest to audit.
    """
    print(f"📊 Auditing RLM Inventory [{config.profile_name.upper()}]...")
    print(f"   Cache: {config.cache_path.name}")

    # 1. Load existing cache
    cache = load_cache(config.cache_path)
    cached_paths: Set[str] = set(cache.keys())

    # 2. Collect live files from manifest
    fs_files = collect_files(config)
    fs_paths: Set[str] = set()
    for f in fs_files:
        try:
            fs_paths.add(str(f.relative_to(PROJECT_ROOT)))
        except ValueError:
            pass

    # 3. Compute coverage deltas
    missing_in_cache = fs_paths - cached_paths
    stale_in_cache = cached_paths - fs_paths
    overlap = fs_paths & cached_paths

    # 4. Print report
    print(f"\n📈 Statistics:")
    print(f"   Files on Disk:    {len(fs_paths)}")
    print(f"   Entries in Cache: {len(cached_paths)}")
    pct = (len(overlap) / len(fs_paths) * 100) if fs_paths else 0
    print(f"   Coverage:         {len(overlap)} / {len(fs_paths)} ({pct:.1f}%)")

    if missing_in_cache:
        print(f"\n❌ Missing from Cache ({len(missing_in_cache)}):")
        for p in sorted(missing_in_cache)[:10]:
            print(f"   - {p}")
        if len(missing_in_cache) > 10:
            print(f"   ... and {len(missing_in_cache) - 10} more.")

    if stale_in_cache:
        print(f"\n⚠️  Stale in Cache ({len(stale_in_cache)}):")
        for p in sorted(stale_in_cache)[:10]:
            print(f"   - {p}")
        if len(stale_in_cache) > 10:
            print(f"   ... and {len(stale_in_cache) - 10} more.")

    if not missing_in_cache and not stale_in_cache:
        print("\n✅ RLM Inventory is perfectly synchronized.")


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main() -> None:
    """Parse CLI arguments and run the audit for the specified profile."""
    parser = argparse.ArgumentParser(description="RLM Inventory — cache coverage audit")
    parser.add_argument("--profile", required=True, help="RLM profile name (from rlm_profiles.json)")

    args = parser.parse_args()

    try:
        config = RLMConfig(profile_name=args.profile)
=======
inventory.py (CLI)
=====================================

Purpose:
    RLM Auditor: Reports coverage of the semantic ledger against the filesystem.
    Specialized for Sanctuary/Legacy Documentation.

Layer: Curate / Rlm

Usage Examples:
    python plugins/rlm-factory/scripts/inventory.py

Supported Object Types:
    - RLM Cache (Sanctuary)

Input Files:
    - .agent/learning/rlm_summary_cache.json
    - Filesystem targets (defined in manifests)

Output:
    - Console report (Statistics, Missing Files, Stale Entries)

Key Functions:
    - audit_inventory(): Logic to compare cache keys against collected file paths.

Script Dependencies:
    - plugins/rlm_factory/scripts/rlm_config.py
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
    from rlm_config import RLMConfig, load_cache, collect_files
except ImportError:
    from tools.rlm_factory.rlm_config import RLMConfig, load_cache, collect_files

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
    parser.add_argument("--type", choices=["project", "tool"], default="project", help="RLM Type (loads manifest from factory)")
    
    args = parser.parse_args()
    
    # Load Config based on Type
    try:
        config = RLMConfig(run_type=args.type)
>>>>>>> origin/main
        audit_inventory(config)
    except Exception as e:
        print(f"❌ Error: {e}")

<<<<<<< HEAD

=======
>>>>>>> origin/main
if __name__ == "__main__":
    main()