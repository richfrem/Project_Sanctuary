#!/usr/bin/env python3
"""
dry-run path updater script for migrating legacy tools/ references to their plugin equivalents.
It strictly uses a dictionary of specific files to ensure active files like tools/cli.py remain untouched.

Usage:
  python3 scripts/update_legacy_paths.py            # Runs in DRY-RUN mode (safe)
  python3 scripts/update_legacy_paths.py --execute  # Actually writes the changes to disk
"""

import os
import argparse
import re
from pathlib import Path

# The exact string mappings to securely target ONLY moved files
EXACT_MAPPINGS = {
    # RLM Factory
    "tools/retrieve/rlm/query_cache.py": "plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py",
    "tools/codify/rlm/distiller.py": "plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py",
    "tools/codify/rlm/rlm_config.py": "plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py",
    "tools/codify/rlm/debug_rlm.py": "plugins/rlm-factory/skills/rlm-curator/scripts/debug_rlm.py",
    "tools/curate/rlm/cleanup_cache.py": "plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py",
    "tools/retrieve/rlm/inventory.py": "plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py",
    "tools/retrieve/rlm/fetch_tool_context.py": "plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py",
    "tools/codify/rlm/refresh_cache.py": "plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py",
    
    # Tool Inventory
    "tools/curate/inventories/manage_tool_inventory.py": "plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py",
    "tools/tool_inventory.json": "plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json",
    ".agent/skills/SKILL.md": "plugins/tool-inventory/skills/tool-inventory/SKILL.md",
    
    # Context Bundler
    "tools/retrieve/bundler/bundle.py": "plugins/context-bundler/scripts/bundle.py",
    "tools/retrieve/bundler/manifest_manager.py": "plugins/context-bundler/scripts/bundle.py",
    
    # Mermaid
    "tools/codify/diagrams/export_mmd_to_image.py": "plugins/mermaid-to-png/skills/convert-mermaid/scripts/convert.py",
    
    # ADR Manager
    "tools/investigate/utils/next_number.py": "plugins/adr-manager/skills/adr-management/scripts/next_number.py",
    
    # Agent Loops / Orchestrator
    "tools/orchestrator/proof_check.py": "plugins/agent-loops/skills/orchestrator/scripts/proof_check.py",
    # We do NOT map tools/orchestrator/workflow_manager.py here because it still exists in tools/
}

# Explicit whitelist of directories and files currently active in `tools/`
# The script will refuse to modify any line containing these references.
WHITELIST = [
    "tools/cli.py",
    "tools/orchestrator/",
    "tools/ai-resources/",
    "tools/README.md",
    "tools/TOOL_INVENTORY.md"
]

# Directories to skip completely for safety/speed
EXCLUDE_DIRS = {
    ".git", ".venv", "node_modules", "archive", "archive-tests", "__pycache__", 
    ".agent/learning", "mcp_servers", "scripts"
}

def scan_and_update(execute=False):
    project_root = Path(__file__).parent.parent.resolve()
    print(f"Starting scan in: {project_root}")
    print(f"Mode: {'EXECUTE (Writing to disk)' if execute else 'DRY-RUN (No changes will be saved)'}\n")
    
    files_modified = 0
    total_replacements = 0
    whitelisted_skips = 0

    for root, dirs, files in os.walk(project_root):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        for file in files:
            if not file.endswith(('.md', '.py', '.sh', '.json', '.yaml', '.yml')):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                continue

            modified = False
            for i, line in enumerate(lines):
                # First check if the line contains a whitelisted active file
                # If so, we intentionally skip touching this line just to be extremely safe
                is_whitelisted = False
                for whitelist_item in WHITELIST:
                    if whitelist_item in line:
                        is_whitelisted = True
                        break
                
                if is_whitelisted:
                    if "tools/" in line:
                        whitelisted_skips += 1
                    continue

                # Process exact mappings
                for old_path, new_path in EXACT_MAPPINGS.items():
                    if old_path in line:
                        lines[i] = lines[i].replace(old_path, new_path)
                        total_replacements += line.count(old_path)
                        modified = True

            if modified:
                rel_path = os.path.relpath(file_path, project_root)
                print(f"[MATCH] {rel_path} -> replacements made.")
                files_modified += 1
                
                if execute:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

    print("\n--- Summary ---")
    print(f"Total files that {'were' if execute else 'would be'} modified: {files_modified}")
    print(f"Total specific path replacements: {total_replacements}")
    print(f"Whitelisted active 'tools/' references skipped for safety: {whitelisted_skips}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safely map legacy tool paths to new plugin paths.")
    parser.add_argument("--execute", action="store_true", help="Apply changes. Without this flag, it performs a dry run.")
    args = parser.parse_args()
    
    scan_and_update(execute=args.execute)
