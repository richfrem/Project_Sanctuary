#!/usr/bin/env python3
"""
Plugin Initializer - Creates a new plugin from template

Usage:
    init_plugin.py <plugin-name> --path <path>

Examples:
    init_plugin.py my-new-plugin --path plugins/
    init_plugin.py custom-tools --path /custom/location
"""

import sys
import json
from pathlib import Path

PLUGIN_JSON_TEMPLATE = {
    "name": "{plugin_name}",
    "description": "[TODO: Brief description of what this plugin does]",
    "version": "1.0.0",
    "author": {
        "name": "[TODO: Author Name]"
    }
}

SKILL_TEMPLATE = """---
name: {skill_name}
description: [TODO: Explanation of what this skill does and when to use it.]
---

# {skill_title}

## Overview

[TODO: Describe what this skill enables]

## Usage

[TODO: usage instructions]
"""

def title_case_name(name):
    """Convert hyphenated name to Title Case."""
    return ' '.join(word.capitalize() for word in name.split('-'))

def init_plugin(plugin_name, path):
    """
    Initialize a new plugin directory with standard structure.

    Args:
        plugin_name: Name of the plugin
        path: Path where the plugin directory should be created

    Returns:
        Path to created plugin directory, or None if error
    """
    # Determine plugin directory path
    plugin_dir = Path(path).resolve() / plugin_name

    # Check if directory already exists
    if plugin_dir.exists():
        print(f"❌ Error: Plugin directory already exists: {plugin_dir}")
        return None

    # Create plugin directory
    try:
        plugin_dir.mkdir(parents=True, exist_ok=False)
        print(f"✅ Created plugin directory: {plugin_dir}")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return None

    # Create .claude-plugin directory and manifest
    try:
        claude_plugin_dir = plugin_dir / '.claude-plugin'
        claude_plugin_dir.mkdir(exist_ok=True)
        
        plugin_json_path = claude_plugin_dir / 'plugin.json'
        manifest_content = json.dumps(PLUGIN_JSON_TEMPLATE, indent=2).replace("{plugin_name}", plugin_name)
        plugin_json_path.write_text(manifest_content)
        print("✅ Created .claude-plugin/plugin.json")
    except Exception as e:
        print(f"❌ Error creating manifest: {e}")
        return None

    # Create skills directory and sample skill
    try:
        skills_dir = plugin_dir / 'skills'
        skills_dir.mkdir(exist_ok=True)
        
        # Create a default skill matching the plugin name
        default_skill_name = plugin_name
        skill_dir = skills_dir / default_skill_name
        skill_dir.mkdir(exist_ok=True)
        
        skill_md_path = skill_dir / 'SKILL.md'
        skill_title = title_case_name(default_skill_name)
        skill_content = SKILL_TEMPLATE.format(
            skill_name=default_skill_name,
            skill_title=skill_title
        )
        skill_md_path.write_text(skill_content)
        print(f"✅ Created sample skill: skills/{default_skill_name}/SKILL.md")
        
    except Exception as e:
        print(f"❌ Error creating skills: {e}")
        return None

    # Print next steps
    print(f"\n✅ Plugin '{plugin_name}' initialized successfully at {plugin_dir}")
    print("\nNext steps:")
    print("1. Edit .claude-plugin/plugin.json to update description and author")
    print(f"2. Edit skills/{default_skill_name}/SKILL.md to define your first skill")
    print("3. Add more skills, agents, or hooks as needed")

    return plugin_dir

def main():
    if len(sys.argv) < 4 or sys.argv[2] != '--path':
        print("Usage: init_plugin.py <plugin-name> --path <path>")
        print("\nExamples:")
        print("  init_plugin.py my-new-plugin --path plugins/")
        sys.exit(1)

    plugin_name = sys.argv[1]
    path = sys.argv[3]

    print(f"🚀 Initializing plugin: {plugin_name}")
    print(f"   Location: {path}")
    print()

    result = init_plugin(plugin_name, path)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
