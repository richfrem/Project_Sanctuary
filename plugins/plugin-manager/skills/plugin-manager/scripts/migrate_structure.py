#!/usr/bin/env python3
"""
Migrate Plugin Structure
========================

Migrates plugins to the standard structure:
- plugins/<plugin>/scripts/*Count -> plugins/<plugin>/skills/<main_skill>/scripts/
- plugins/<plugin>/docs/* -> plugins/<plugin>/skills/<main_skill>/references/

Usage:
    python3 plugins/tool-inventory/skills/tool-inventory/scripts/migrate_structure.py [--apply]
"""

import sys
import shutil
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"

def get_main_skill(plugin_path: Path) -> Path:
    """
    Heuristic to find the 'main' skill directory.
    If multiple, picks the one matching the plugin name, or the first one.
    If none, creates one matching the plugin name.
    """
    skills_dir = plugin_path / "skills"
    if not skills_dir.exists():
        skills_dir.mkdir(parents=True, exist_ok=True)
        
    # specific overrides or heuristics
    potential_skills = [d for d in skills_dir.iterdir() if d.is_dir()]
    
    if not potential_skills:
        # Create default skill matching plugin name
        main_skill = skills_dir / plugin_path.name
        main_skill.mkdir(exist_ok=True)
        return main_skill
        
    # Try to find match
    for s in potential_skills:
        if s.name == plugin_path.name:
            return s
            
    # Default to first
    return potential_skills[0]

def migrate_plugin(plugin_path: Path, dry_run: bool = True):
    print(f"\n📦 Processing {plugin_path.name}...")
    
    main_skill = get_main_skill(plugin_path)
    print(f"   Target Skill: {main_skill.name}")
    
    # 1. Migrate Scripts
    src_scripts = plugin_path / "scripts"
    tgt_scripts = main_skill / "scripts"
    
    if src_scripts.exists() and any(src_scripts.iterdir()):
        print(f"   - Found top-level 'scripts/'")
        if not dry_run:
            tgt_scripts.mkdir(exist_ok=True)
            for item in src_scripts.iterdir():
                if item.name == "__pycache__": continue
                dst = tgt_scripts / item.name
                if dst.exists():
                    print(f"     ⚠️  Conflict: {dst} already exists. Skipping.")
                else:
                    print(f"     Moving {item.name} -> {tgt_scripts.relative_to(plugin_path)}")
                    shutil.move(str(item), str(dst))
            
            # Clean up if empty
            if not any(src_scripts.iterdir()):
                src_scripts.rmdir()
                print("     Removed empty 'scripts/' directory")
    
    # 2. Migrate Docs -> Reference
    src_docs = plugin_path / "docs"
    tgt_refs = main_skill / "references"
    
    if src_docs.exists() and any(src_docs.iterdir()):
        print(f"   - Found top-level 'docs/'")
        if not dry_run:
            tgt_refs.mkdir(exist_ok=True)
            for item in src_docs.iterdir():
                dst = tgt_refs / item.name
                if dst.exists():
                    print(f"     ⚠️  Conflict: {dst} already exists. Skipping.")
                else:
                    print(f"     Moving {item.name} -> {tgt_refs.relative_to(plugin_path)}")
                    shutil.move(str(item), str(dst))
                    
            # Clean up if empty
            if not any(src_docs.iterdir()):
                src_docs.rmdir()
                print("     Removed empty 'docs/' directory")

def main():
    parser = argparse.ArgumentParser(description="Migrate Plugin Structure")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()
    
    if not args.apply:
        print("🔍 DRY RUN MODE (Use --apply to execute)\n")
    
    for plugin_path in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_path.is_dir(): continue
        if plugin_path.name.startswith("."): continue
        # tool-inventory is already done manually
        if plugin_path.name == "tool-inventory": continue
        
        migrate_plugin(plugin_path, dry_run=not args.apply)

if __name__ == "__main__":
    main()
