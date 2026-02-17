#!/usr/bin/env python3
"""
Sync Workflows Bridge
---------------------
Synchronizes custom workflow definitions from .agent/workflows/ to
other agent configuration directories (Claude, Copilot).

Source:
  - .agent/workflows/sanctuary_protocols/
  - .agent/workflows/utilities/

Targets:
  - .claude/commands/
  - .github/prompts/
  - .gemini/commands/ (converted to TOML)

Note:
  - .github/workflows/ is reserved for GitHub Actions and is NOT touched.
  - .gemini/ reads .agent/workflows/ natively (no sync needed).

Usage:
    python3 plugins/spec-kitty/sync_workflows.py
"""
import os
import shutil
import sys
import argparse
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE_WORKFLOWS_DIR = PROJECT_ROOT / ".agent" / "workflows"

# Target Agent Directories
TARGET_AGENTS = {
    "CLAUDE":  PROJECT_ROOT / ".claude" / "commands",
    "COPILOT": PROJECT_ROOT / ".github" / "prompts", 
    "GEMINI":  PROJECT_ROOT / ".gemini" / "commands",
}

def transform_content(content, filename, agent_name):
    """Transforms workflow content based on agent requirements."""
    stem = Path(filename).stem
    
    if agent_name == "GEMINI":
        # Wrap in TOML
        # Simple description extraction or fallback
        description = stem.replace("-", " ").title()
        return f'description = "{description}"\nprompt = """\n{content}\n"""\n', f"{stem}.toml"
    
    elif agent_name == "COPILOT":
        # Rename to .prompt.md, keep content
        return content, f"{stem}.prompt.md"
        
    return content, filename

def sync_recursive(source_dir, target_dir, agent_name):
    """Recursively syncs and transforms files."""
    for item in source_dir.iterdir():
        if item.name.startswith(".") or item.name.startswith("spec-kitty"):
            continue
            
        if item.is_dir():
            # Recurse
            new_target = target_dir / item.name
            new_target.mkdir(parents=True, exist_ok=True)
            sync_recursive(item, new_target, agent_name)
            
        elif item.is_file() and item.suffix == ".md":
            content = item.read_text(encoding="utf-8")
            new_content, new_filename = transform_content(content, item.name, agent_name)
            
            target_path = target_dir / new_filename
            target_path.write_text(new_content, encoding="utf-8")
            print(f"  ✓ Synced: {new_filename}")

def sync_workflows():
    print("🔄 Syncing Custom Workflows...")
    
    if not SOURCE_WORKFLOWS_DIR.exists():
        print(f"❌ Error: Source workflows directory not found: {SOURCE_WORKFLOWS_DIR}")
        sys.exit(1)

    for agent_name, target_root in TARGET_AGENTS.items():
        print(f"--- Syncing to {agent_name} ---")
        
        try:
            target_root.mkdir(parents=True, exist_ok=True)
            # Sync
            sync_recursive(SOURCE_WORKFLOWS_DIR, target_root, agent_name)
        except Exception as e:
            print(f"  ⚠️  Error syncing to {agent_name}: {e}")

    print(f"\n✅ Workflow Sync Complete.")




def main():
    parser = argparse.ArgumentParser(description="Sync custom workflows to native directories.")
    parser.add_argument("--all", action="store_true", help="Sync workflows to agent directories")
    args = parser.parse_args()

    if not args.all:
        print("Usage: provide --all to sync workflows")
        sys.exit(1)

    sync_workflows()

if __name__ == "__main__":
    main()
