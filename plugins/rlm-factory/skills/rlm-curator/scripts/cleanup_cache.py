#!/usr/bin/env python3
"""
<<<<<<< HEAD
cleanup_cache.py
=====================================

Purpose:
    RLM Cleanup: Removes stale and orphan entries from the RLM semantic ledger.
    Supports dry-run mode by default; requires --apply to commit changes.

Layer: Curate / Rlm

Usage:
    # Dry run — preview stale entries
    python plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py --profile plugins

    # Remove files whose source no longer exists
    python plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py --profile plugins --apply

    # Remove entries outside the manifest + failed distillations
    python plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py --profile tools --prune-orphans --prune-failed --apply

Related:
    - rlm_config.py (configuration & utilities)
    - inventory.py (coverage audit)
    - distiller.py (cache population)
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Set

# ============================================================
# PATHS
# File is at: plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py
# Root is 6 levels up (scripts→rlm-curator→skills→rlm-factory→plugins→ROOT)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from rlm_config import RLMConfig, load_cache, save_cache, collect_files
except ImportError as e:
    print(f"❌ Could not import rlm_config: {e}")
    sys.exit(1)


# ----------------------------------------------------------
# run_cleanup — entry-by-entry sweep of the cache
# ----------------------------------------------------------
def run_cleanup(
    config: RLMConfig,
    apply: bool,
    prune_orphans: bool,
    prune_failed: bool,
    verbose: bool
) -> int:
    """
    Scan the cache for stale, orphan, and failed entries, then optionally remove them.

    Three removal criteria (controlled by flags):
    1. **Stale**: Source file no longer exists on disk (always checked).
    2. **Failed**: Summary is the sentinel string `[DISTILLATION FAILED]`.
    3. **Orphan**: File exists but is not covered by the profile's manifest.

    Args:
        config: Active RLMConfig defining the cache and manifest.
        apply: If True, write the pruned cache to disk. If False, dry-run only.
        prune_orphans: Include orphan entries (not in manifest) in removals.
        prune_failed: Include entries with failed distillations in removals.
        verbose: Print per-entry status during the scan.

    Returns:
        Number of entries removed (or that would be removed in dry-run mode).
    """
    print(f"🧹 Checking cache [{config.profile_name.upper()}]: {config.cache_path.name}")

    if not config.cache_path.exists():
        print("   Cache file not found. Nothing to clean.")
        return 0

    cache: Dict = load_cache(config.cache_path)
    print(f"   Entries in cache: {len(cache)}")

    entries_to_remove = []
    authorized_files: Optional[Set[str]] = None

    for rel_path, entry in list(cache.items()):
        full_path = PROJECT_ROOT / rel_path

        # 1. Failed distillation check
        if prune_failed and entry.get("summary") == "[DISTILLATION FAILED]":
            entries_to_remove.append(rel_path)
            if verbose:
                print(f"   [FAILED]  {rel_path}")
            continue

        # 2. Stale check — source file missing from disk
        if not full_path.exists():
            entries_to_remove.append(rel_path)
            if verbose:
                print(f"   [MISSING] {rel_path}")
            continue

        # 3. Orphan check — file exists but is not in manifest
        if prune_orphans:
            if authorized_files is None:
                print("   Building authorized file list from manifest...")
                authorized_files = {str(f.resolve()) for f in collect_files(config)}
                print(f"   Authorized files: {len(authorized_files)}")

            if str(full_path.resolve()) not in authorized_files:
                entries_to_remove.append(rel_path)
                if verbose:
                    print(f"   [ORPHAN]  {rel_path}")
                continue

        if verbose:
            print(f"   [OK]      {rel_path}")

    count = len(entries_to_remove)
    print(f"   Entries to remove: {count}")

    if count == 0:
        print("   ✅ Cache is clean.")
        return 0

    if apply:
        for key in entries_to_remove:
            del cache[key]
        save_cache(cache, config.cache_path)
        print(f"   ✅ Removed {count} entries.")
    else:
        print(f"\n   DRY RUN: Would remove {count} entries. Re-run with --apply to commit.")

    return count


# ----------------------------------------------------------
# remove_entry — programmatic single-entry removal API
# ----------------------------------------------------------
def remove_entry(profile_name: str, file_path: str) -> bool:
    """
    Programmatic API to remove a single entry from a profile's cache.

    Normalizes path separators before performing the lookup to ensure
    cross-platform compatibility.

    Args:
        profile_name: Name of the RLM profile whose cache to modify.
        file_path: Relative path of the cache key to remove.

    Returns:
        True if the entry was found and removed, False otherwise.
    """
    try:
        config = RLMConfig(profile_name=profile_name)
        if not config.cache_path.exists():
            return False
        cache = load_cache(config.cache_path)
        norm_path = file_path.replace("\\", "/")
        if norm_path in cache:
            del cache[norm_path]
            save_cache(cache, config.cache_path)
            print(f"🗑️  [RLM] Removed '{norm_path}' from '{profile_name}' cache.")
            return True
        print(f"⚠️  [RLM] Entry not found: {file_path}")
        return False
    except Exception as e:
        print(f"❌ [RLM] Error: {e}")
        return False


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main() -> None:
    """Parse CLI arguments and dispatch to run_cleanup()."""
    parser = argparse.ArgumentParser(
        description="RLM Cleanup — prune stale, orphan, and failed cache entries"
    )
    parser.add_argument("--profile", required=True, help="RLM profile name (from rlm_profiles.json)")
    parser.add_argument("--apply", action="store_true", help="Commit the removal (default is dry run)")
    parser.add_argument("--prune-orphans", action="store_true", help="Remove entries not covered by the manifest")
    parser.add_argument("--prune-failed", action="store_true", help="Remove entries with [DISTILLATION FAILED]")
    parser.add_argument("--v", action="store_true", help="Verbose per-entry logging")

    args = parser.parse_args()
    config = RLMConfig(profile_name=args.profile)
    run_cleanup(
        config,
        apply=args.apply,
        prune_orphans=args.prune_orphans,
        prune_failed=args.prune_failed,
        verbose=args.v
    )


=======
cleanup_cache.py (CLI)
=====================================

Purpose:
    RLM Cleanup: Removes stale and orphan entries from the Recursive Language Model ledger.

Layer: Curate / Rlm

Usage Examples:
    python plugins/rlm-factory/scripts/cleanup_cache.py --help
    python plugins/rlm-factory/scripts/cleanup_cache.py --apply --prune-orphans

Supported Object Types:
    - Generic

CLI Arguments:
    --apply         : Perform the deletion
    --prune-orphans : Remove entries not matching manifest
    --v             : Verbose mode

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - load_manifest_globs(): Load include/exclude patterns from manifest.
    - matches_any(): Check if path matches any glob pattern or is inside a listed directory.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path to find tools package
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from rlm_config import RLMConfig, load_cache, save_cache, should_skip
except ImportError:
    try:
        from tools.tool_inventory.rlm_config import RLMConfig, load_cache, save_cache, should_skip
    except ImportError:
        print("❌ Could not import RLMConfig (tried local and tools.tool_inventory)")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Clean up RLM cache.")
    # parser.add_argument("--type", choices=["project", "tool"], default="project", help="RLM Type (loads manifest from factory)")
    parser.add_argument("--apply", action="store_true", help="Perform the deletion")
    parser.add_argument("--prune-orphans", action="store_true", help="Remove entries not matching manifest")
    parser.add_argument("--prune-failed", action="store_true", help="Remove entries with [DISTILLATION FAILED]")
    parser.add_argument("--v", action="store_true", help="Verbose mode")
    args = parser.parse_args()
    
    # Load Config based on Type
    config = RLMConfig(run_type="project")

    print(f"Checking cache at: {config.cache_path}")
    
    if not config.cache_path.exists():
        print("Cache file not found.")
        return

    cache = load_cache(config.cache_path)

    if args.prune_orphans:
        print(f"Loaded configuration for [{config.type.upper()}] with parser: {config.parser_type}")

    initial_count = len(cache)
    print(f"Total entries in cache: {initial_count}")

    # The Logic
    entries_to_remove = []
    authorized_files = None
    
    for relative_path, entry in list(cache.items()):
        full_path = PROJECT_ROOT / relative_path
        
        # 1. Check Distillation Failure (Explicit request)
        if args.prune_failed:
            summary = entry.get("summary", "")
            if summary == "[DISTILLATION FAILED]":
                entries_to_remove.append(relative_path)
                if args.v:
                    print(f"  [FAILED] {relative_path}")
                continue

        # 2. Check existence (Stale)
        if not full_path.exists():
            entries_to_remove.append(relative_path)
            if args.v:
                print(f"  [MISSING] {relative_path}")
            continue

        # 3. Check manifest (Orphan)
        if args.prune_orphans:
            # STRICT ORPHAN CHECK:
            # If the file is not in the list of files matched by the configuration (Manifest/Inventory),
            # it is an orphan.
            
            # Lazy load authorized set on first use
            if authorized_files is None:
                print("Building authorized file list from manifest...")
                # We need to import collect_files from rlm_config
                try:
                    from rlm_config import collect_files
                except ImportError:
                    from tools.tool_inventory.rlm_config import collect_files
                files = collect_files(config)
                # Store as set of resolved strings for fast lookup
                authorized_files = set(str(f.resolve()) for f in files)
                print(f"Authorized files count: {len(authorized_files)}")

            try:
                # Resolve cache path to absolute for comparison
                full_path_str = str(full_path.resolve())
                
                if full_path_str not in authorized_files:
                    entries_to_remove.append(relative_path)
                    if args.v:
                        print(f"  [ORPHAN] {relative_path} (Not in manifest)")
                    continue
            except Exception as e:
                # If we can't resolve, it might be a bad path, safety remove? 
                # Or keep safe. Let's log.
                if args.v: print(f"  [ERROR] resolving {relative_path}: {e}")
                continue

        if args.v:
           print(f"  [OK] {relative_path}")

    remove_count = len(entries_to_remove)
    print(f"Entries to remove: {remove_count}")

    if remove_count == 0:
        print("Cache is clean. No action needed.")
        return

    if args.apply:
        print(f"Removing {remove_count} entries...")
        for key in entries_to_remove:
            if key in cache:
                del cache[key]
        
        save_cache(cache, config.cache_path)
        print("Cache updated successfully.")
    else:
        print("\nDRY RUN COMPLETE.")
        print(f"Found {remove_count} entries to remove (Stale + Orphans).")
        print("To actually remove these entries, run:")
        if args.prune_orphans:
            print(f"  python plugins/rlm-factory/scripts/cleanup_cache.py --apply --prune-orphans")
        else:
            print(f"  python plugins/rlm-factory/scripts/cleanup_cache.py --apply")

def remove_entry(run_type: str, file_path: str) -> bool:
    """
    Programmatic API to remove a single entry from the cache.
    Args:
        run_type: 'legacy' or 'tool'
        file_path: Relative path to the file (e.g. tools/cli.py)
    Returns:
        True if removed, False if not found or error.
    """
    try:
        config = RLMConfig(run_type=run_type)
        if not config.cache_path.exists():
            return False
            
        cache = load_cache(config.cache_path)
        
        # Normalize keys
        target_keys = [
            file_path, 
            file_path.replace('\\', '/'),
            str(Path(file_path)) 
        ]
        
        found_key = None
        for k in cache.keys():
            if k in target_keys:
                found_key = k
                break
        
        if found_key:
            del cache[found_key]
            save_cache(cache, config.cache_path)
            print(f"🗑️  [RLM] Removed {found_key} from {run_type} cache.")
            return True
        else:
             print(f"⚠️  [RLM] Entry not found in cache: {file_path}")
             return False

    except Exception as e:
        print(f"❌ [RLM] Error removing {file_path}: {e}")
        return False

>>>>>>> origin/main
if __name__ == "__main__":
    main()
