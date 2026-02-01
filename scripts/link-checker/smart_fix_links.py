#!/usr/bin/env python3
"""
smart_fix_links.py (CLI)
=====================================

Purpose:
    Auto-repair utility for broken Markdown links.
    Uses a file inventory to find the correct location of moved or renamed files
    and updates the links in-place. Supports "fuzzy" matching for ambiguous cases.

Layer: Curate / Link Checker

Usage Examples:
    python scripts/link-checker/smart_fix_links.py --dry-run
    python scripts/link-checker/smart_fix_links.py

CLI Arguments:
    --dry-run       : Report proposed changes without modifying files (Safety Mode)

Input Files:
    - scripts/link-checker/file_inventory.json (Source of Truth)
    - **/*.md (Target files to fix)

Output:
    - Modified .md files
    - Console report of fixes

Key Functions:
    - fix_links_in_file(): regex replacement logic with inventory lookup.

Script Dependencies:
    - mcp_servers/lib/exclusion_config.py

Consumed by:
    - /post-move-link-check (Workflow)
    - Manual maintenance
"""
import os
import json
import re
import argparse
import sys
from pathlib import Path
from urllib.parse import unquote

# Add mcp_servers to path for shared config
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp_servers"))

# Import shared exclusion config from mcp_servers/lib
from lib.exclusion_config import EXCLUDE_DIR_NAMES, ALWAYS_EXCLUDE_FILES


def find_project_root() -> Path:
    """Find the Project Sanctuary root by looking for markers."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "IDENTITY").exists() or (current / "ADRs").exists():
            return current
        current = current.parent
    return Path.cwd()


def should_skip_file(filename: str) -> bool:
    """Check if file should be skipped based on exclusion config."""
    for excl in ALWAYS_EXCLUDE_FILES:
        if isinstance(excl, str):
            if filename == excl or filename.endswith(f'/{excl}'):
                return True
        elif hasattr(excl, 'match'):  # Compiled regex
            if excl.match(filename):
                return True
    return False


def load_inventory(inventory_path: Path) -> dict:
    """Load the file inventory JSON."""
    with open(inventory_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_relative_path(start_file: Path, target_path: Path) -> str:
    """Calculate relative path from start_file to target_path."""
    start_dir = start_file.parent
    return os.path.relpath(target_path, start_dir).replace('\\', '/')


def fix_links_in_file(file_path: Path, inventory: dict, root_dir: Path, dry_run: bool = False) -> list:
    """Fix broken links in a single file. Returns list of fixes made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return []

    original_content = content
    fixes = []
    
    # Regex for standard markdown links [label](path)
    link_pattern = re.compile(r'(\[.*?\])\((.*?)\)')
    
    def replace_link(match):
        label = match.group(1)
        original_link_path = match.group(2)
        
        # Skip web links/anchors
        if original_link_path.startswith(('http', 'mailto:', '#', 'file://')):
            return match.group(0)
            
        clean_link = original_link_path.split('#')[0]
        anchor = '#' + original_link_path.split('#')[1] if '#' in original_link_path else ''
        
        # Verify if currently valid
        file_dir = file_path.parent
        current_abs_target = file_dir / unquote(clean_link)
        
        if current_abs_target.exists():
            return match.group(0)  # Valid, don't touch
            
        # It's broken. Find the basename.
        basename = Path(unquote(clean_link)).name
        
        # Lookup in inventory
        candidates = inventory.get(basename, [])
        
        if not candidates:
            # Try case-insensitive lookup
            lower_basename = basename.lower()
            for k, v in inventory.items():
                if k.lower() == lower_basename:
                    candidates = v
                    break
        
        if not candidates:
            # File truly missing - mark it
            fixes.append(f"MISSING: {basename} in {file_path.name}")
            return match.group(0)  # Leave unchanged
            
        if len(candidates) == 1:
            # Unique match! Fix it.
            new_target = root_dir / candidates[0]
            new_rel_path = calculate_relative_path(file_path, new_target)
            fixes.append(f"FIXED: {basename} -> {new_rel_path}")
            return f"{label}({new_rel_path}{anchor})"
            
        # Ambiguous - try to find best match by path similarity
        original_parts = set(Path(clean_link).parts)
        best_match = None
        best_score = 0
        for candidate in candidates:
            candidate_parts = set(Path(candidate).parts)
            score = len(original_parts & candidate_parts)
            if score > best_score:
                best_score = score
                best_match = candidate
                
        if best_match and best_score > 0:
            new_target = root_dir / best_match
            new_rel_path = calculate_relative_path(file_path, new_target)
            fixes.append(f"FIXED (best match): {basename} -> {new_rel_path}")
            return f"{label}({new_rel_path}{anchor})"
        
        fixes.append(f"AMBIGUOUS: {basename} has {len(candidates)} candidates")
        return match.group(0)

    new_content = link_pattern.sub(replace_link, content)
    
    if new_content != original_content and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
    return fixes


def main():
    parser = argparse.ArgumentParser(description="Smart Link Fixer for Project Sanctuary")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report")
    args = parser.parse_args()
    
    root_dir = find_project_root()
    inventory_path = Path(__file__).parent / 'file_inventory.json'
    
    if not inventory_path.exists():
        print("Inventory not found. Run map_repository_files.py first.")
        return

    inventory = load_inventory(inventory_path)
    
    print(f"{'DRY RUN: ' if args.dry_run else ''}Fixing links in {root_dir}...")
    
    total_fixes = 0
    files_modified = 0
    
    for root, dirs, files in os.walk(root_dir):
        # Use shared exclusion config
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        
        for filename in files:
            if not filename.endswith('.md'):
                continue
            if should_skip_file(filename):
                continue
                
            file_path = Path(root) / filename
            fixes = fix_links_in_file(file_path, inventory, root_dir, args.dry_run)
            
            if fixes:
                rel_path = file_path.relative_to(root_dir)
                print(f"\n{rel_path}:")
                for fix in fixes:
                    print(f"  {fix}")
                total_fixes += len(fixes)
                files_modified += 1
                    
    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n{action} {files_modified} files with {total_fixes} link changes.")


if __name__ == "__main__":
    main()