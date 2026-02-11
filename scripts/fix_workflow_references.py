#!/usr/bin/env python3
"""
Fix Workflow References Script
================================
Phase 1: Update full path references from old to new locations.

Excludes workflow_migration_map_*.json files to preserve the mapping history.
"""

import json
import os
import sys

def main():
    # Load migration map (v1 only - contains original->new mappings)
    map_file = ".agent/workflow_migration_map_v1.json"
    
    if not os.path.exists(map_file):
        print(f"Error: Map file not found: {map_file}")
        sys.exit(1)
    
    print(f"Loading map: {map_file}")
    with open(map_file, "r") as f:
        migration_map = json.load(f)
    
    # Phase 1: Full path replacements
    # Phase 2: Bare filename replacements (for references without paths)
    # Phase 3: Slash-command style replacements (e.g., /workflow-seal)
    replacements = {}
    
    for old_path, new_path in migration_map.items():
        if not new_path:
            continue
        
        # Phase 1: Full path replacement
        replacements[old_path] = new_path
        
        # Phase 2: Bare filename replacement
        old_filename = os.path.basename(old_path)
        new_filename = os.path.basename(new_path)
        
        # Only add if filenames are different to avoid redundant replacements
        if old_filename != new_filename:
            replacements[old_filename] = new_filename
            
            # Phase 3: Slash-command style (strip .md extension and add /)
            old_cmd = "/" + old_filename.replace(".md", "")
            new_cmd = "/" + new_filename.replace(".md", "")
            replacements[old_cmd] = new_cmd
    
    print(f"Loaded {len(replacements)} replacement rules.")
    
    # Walk the directory
    extensions = {".md", ".mmd", ".py", ".json", ".js", ".ts", ".txt"}
    
    count = 0
    for root, dirs, files in os.walk("."):
        # Skip certain directories
        if ".git" in dirs:
            dirs.remove(".git")
        if "node_modules" in dirs:
            dirs.remove("node_modules")
        if ".venv" in dirs:
            dirs.remove(".venv")
        
        for file in files:
            # CRITICAL: Don't edit the migration map files themselves
            if file.startswith("workflow_migration_map"):
                continue
            if file == "fix_workflow_references.py":
                continue
            
            ext = os.path.splitext(file)[1]
            if ext not in extensions and file != "Taskfile":
                continue
            
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                original_content = content
                
                # Replace all occurrences
                for old_path, new_path in replacements.items():
                    if old_path in content:
                        content = content.replace(old_path, new_path)
                
                # Only write if changes were made
                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Updated: {file_path}")
                    count += 1
                    
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    
    print(f"\nDone. Updated {count} files.")

if __name__ == "__main__":
    main()
