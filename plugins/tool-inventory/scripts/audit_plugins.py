#!/usr/bin/env python3
"""
Audit Plugin Inventory
======================

Audits the `tool_inventory.json` against the actual file system to ensure all
scripts in `plugins/` are registered.

Checks for:
1. Missing Scripts: Files in `plugins/` not in inventory.
2. Orphan Entries: Inventory entries pointing to non-existent files.
3. RLM Sync: Checks if tools are present in the RLM cache.

Usage:
    python3 plugins/tool-inventory/scripts/audit_plugins.py
"""

import sys
import json
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
INVENTORY_PATH = PROJECT_ROOT / "tools" / "tool_inventory.json"
RLM_CACHE_PATH = PROJECT_ROOT / ".agent" / "learning" / "rlm_tool_cache.json"

def load_json(path):
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def main():
    print(f"🔍 Auditing Plugin Inventory...")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Inventory:    {INVENTORY_PATH}")
    if not INVENTORY_PATH.exists():
        print(f"❌ Error: Inventory file not found at {INVENTORY_PATH}")
        sys.exit(1)
        
    inventory = load_json(INVENTORY_PATH)
    rlm_cache = load_json(RLM_CACHE_PATH)
    
    # 1. Map Inventory
    inventory_paths = set()
    scripts_dict = inventory.get("scripts", {})
    for category, tools in scripts_dict.items():
        for tool in tools:
            inventory_paths.add(tool.get("path"))
            
    print(f"   Loaded Inventory: {len(inventory_paths)} unique paths") # Debug
        
    # 2. Map File System (Plugins only)
    plugins_dir = PROJECT_ROOT / "plugins"
    args_files = set()
    
    print(f"   Scanning {plugins_dir}...")
    for file_path in plugins_dir.rglob("*.py"):
        # Filters
        if file_path.name == "__init__.py": continue
        if "tests" in file_path.parts: continue
        if "node_modules" in file_path.parts: continue
        if ".venv" in file_path.parts: continue
        if "__pycache__" in file_path.parts: continue
        
        try:
            rel_path = str(file_path.relative_to(PROJECT_ROOT))
            args_files.add(rel_path)
        except ValueError:
            continue

    # 3. Analyze
    missing_in_inventory = args_files - inventory_paths
    orphans_in_inventory = {p for p in inventory_paths if p.startswith("plugins/") and p not in args_files} # Only check plugins
    
    # Check RLM Coverage
    rlm_keys = set(rlm_cache.keys())
    missing_in_rlm = {p for p in args_files if p not in rlm_keys}

    print("\n" + "="*50)
    print("📊 Audit Results")
    print("="*50)
    
    print(f"Total Plugin Scripts: {len(args_files)}")
    print(f"Inventory Entries:    {len(inventory_paths)}")
    
    if missing_in_inventory:
        print(f"\n❌ Missing from Inventory ({len(missing_in_inventory)}):")
        for p in sorted(missing_in_inventory):
            print(f"   - {p}")
    else:
        print("\n✅ All scripts registered in inventory.")

    if orphans_in_inventory:
        # Double check if file actually exists (maybe filter logic was too strict?)
        confirmed_orphans = []
        for p in orphans_in_inventory:
            if not (PROJECT_ROOT / p).exists():
                confirmed_orphans.append(p)
        
        if confirmed_orphans:
            print(f"\n⚠️  Orphan Inventory Entries ({len(confirmed_orphans)}):")
            for p in sorted(confirmed_orphans):
                print(f"   - {p}")
    
    if missing_in_rlm:
        print(f"\n⚠️  Missing from RLM Cache ({len(missing_in_rlm)}):")
        for p in sorted(missing_in_rlm):
            print(f"   - {p}")
            
    if not missing_in_inventory and not orphans_in_inventory and not missing_in_rlm:
        print("\n✨ Perfect System State!")
        sys.exit(0)
    else:
        print("\n⚠️  Issues Found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
