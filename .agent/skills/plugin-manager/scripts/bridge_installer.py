#!/usr/bin/env python3
"""
Plugin Bridge Installer
=======================

Installs Agent Plugins (.claude-plugin structure) into target environments.

Supported Targets:
- Antigravity (.agent/)
- GitHub Copilot (.github/)
- Gemini (.gemini/)

Usage:
  python3 plugins/plugin-manager/scripts/bridge_installer.py --plugin <path> [--target <auto|antigravity|github|gemini>]
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path

# --- Constants ---

TARGET_MAPPINGS = {
    "antigravity": {
        "check": ".agent",
        "workflows": ".agent/workflows",
        "skills": ".agent/skills",
        "rules": ".agent/rules",
        "tools": "tools"
    },
    "github": {
        "check": ".github",
        "workflows": ".github/prompts",
        "instructions": ".github/copilot-instructions.md"
    },
    "gemini": {
        "check": ".gemini",
        "workflows": ".gemini/commands"
    },
    "claude": {
        "check": ".claude",
        "commands": ".claude/commands"
    }
}

# --- Core Logic ---

def transform_content(content: str, target_agent: str) -> str:
    """Transforms content for specific target agents."""
    # 1. Actor Swapping (Spec Kitty convention)
    # Replace default actor with target
    if target_agent == "antigravity":
        content = content.replace('--actor "windsurf"', '--actor "antigravity"')
        content = content.replace('--actor "claude"', '--actor "antigravity"')
    elif target_agent == "github":
        content = content.replace('--actor "windsurf"', '--actor "copilot"')
        content = content.replace('--actor "claude"', '--actor "copilot"')
    elif target_agent == "gemini":
        content = content.replace('--actor "windsurf"', '--actor "gemini"')
        content = content.replace('--actor "claude"', '--actor "gemini"')
        content = content.replace('$ARGUMENTS', '{{args}}') # Gemini argument syntax
    elif target_agent == "claude":
        content = content.replace('--actor "windsurf"', '--actor "claude"')
        # No change needed if already "claude"

    return content

def detect_targets(root: Path):
    targets = []
    for name, config in TARGET_MAPPINGS.items():
        if (root / config["check"]).exists():
            targets.append(name)
    return targets

def install_antigravity(plugin_path: Path, root: Path, metadata: dict):
    print("  [Antigravity] Installing...")
    target_wf = root / TARGET_MAPPINGS["antigravity"]["workflows"]
    target_skills = root / TARGET_MAPPINGS["antigravity"]["skills"]
    target_tools = root / TARGET_MAPPINGS["antigravity"]["tools"]

    target_wf.mkdir(parents=True, exist_ok=True)
    target_skills.mkdir(parents=True, exist_ok=True)
    target_tools.mkdir(parents=True, exist_ok=True)

    plugin_name = metadata.get("name", plugin_path.name)

    # 1. Workflows (Commands)
    commands_dir = plugin_path / "commands"
    if not commands_dir.exists():
        commands_dir = plugin_path / "workflows"
        
    if commands_dir.exists():
        plugin_wf_dir = target_wf / plugin_name
        plugin_wf_dir.mkdir(parents=True, exist_ok=True)
        for f in commands_dir.glob("*.md"):
            content = f.read_text()
            content = transform_content(content, "antigravity")
            dest = plugin_wf_dir / f"{plugin_name}_{f.name}" # Namespace conflict prevention
            dest.write_text(content)
            print(f"    -> Workflow: {dest.relative_to(root)}")

    # 2. Skills
    skills_dir = plugin_path / "skills"
    if skills_dir.exists():
        shutil.copytree(skills_dir, target_skills, dirs_exist_ok=True)
        print(f"    -> Skills: {target_skills.relative_to(root)}")

    # 3. Tools / Scripts (DEPRECATED: Direct execution from plugins/ preferred)
    # scripts_dir = plugin_path / "scripts"
    # if scripts_dir.exists():
    #     # Copy to tools/{plugin_name}/
    #     dest_tools = target_tools / plugin_name
    #     if dest_tools.exists(): shutil.rmtree(dest_tools) 
    #     # shutil.copytree(scripts_dir, dest_tools)
    #     # print(f"    -> Tools: {dest_tools.relative_to(root)} (DEPRECATED MIRROR)")

    # 4. Resources (Manifests, Prompts, Configs)
    # DEPRECATED: Resources now live in plugins/<plugin>/resources and are accessed directly.
    # No copy to tools/ needed.
    # resources_dir = plugin_path / "resources"
    # if resources_dir.exists():
    #    print(f"    -> Resources: {resources_dir.relative_to(root)} (Referenced in-place)")

def install_github(plugin_path: Path, root: Path, metadata: dict):
    print("  [GitHub] Installing...")
    target_prompts = root / TARGET_MAPPINGS["github"]["workflows"]
    target_prompts.mkdir(parents=True, exist_ok=True)

    plugin_name = metadata.get("name", plugin_path.name)

    # 1. Workflows -> Prompts
    commands_dir = plugin_path / "commands"
    if not commands_dir.exists():
        commands_dir = plugin_path / "workflows"
        
    if commands_dir.exists():
        for f in commands_dir.glob("*.md"):
            content = f.read_text()
            content = transform_content(content, "github")
            # Rename .md -> .prompt.md
            dest = target_prompts / f"{plugin_name}_{f.stem}.prompt.md"
            dest.write_text(content)
            print(f"    -> Prompt: {dest.relative_to(root)}")

    # 2. Instructions -> Copilot Instructions
    # Append summary of skills? (Simplified for now: skip)
    print("    -> Skills: Not natively supported by Copilot yet (skipped).")

def install_gemini(plugin_path: Path, root: Path, metadata: dict):
    print("  [Gemini] Installing...")
    target_cmds = root / TARGET_MAPPINGS["gemini"]["workflows"]
    target_cmds.mkdir(parents=True, exist_ok=True)

    plugin_name = metadata.get("name", plugin_path.name)

    # 1. Workflows -> TOML Commands
    commands_dir = plugin_path / "commands"
    if not commands_dir.exists():
        commands_dir = plugin_path / "workflows"
        
    if commands_dir.exists():
        for f in commands_dir.glob("*.md"):
            content = f.read_text()
            content = transform_content(content, "gemini")
            # Wrap in TOML
            toml_content = f'command = "{plugin_name}:{f.stem}"\ndescription = "Imported from plugin"\nprompt = """\n{content}\n"""'
            dest = target_cmds / f"{plugin_name}_{f.stem}.toml"
            dest.write_text(toml_content)
            print(f"    -> Command: {dest.relative_to(root)}")


def install_claude(plugin_path: Path, root: Path, metadata: dict):
    print("  [Claude] Installing...")
    target_cmds = root / TARGET_MAPPINGS["claude"]["commands"]
    target_cmds.mkdir(parents=True, exist_ok=True)

    plugin_name = metadata.get("name", plugin_path.name)

    # 1. Workflows (Commands)
    commands_dir = plugin_path / "commands"
    if not commands_dir.exists():
        commands_dir = plugin_path / "workflows"
        
    if commands_dir.exists():
        for f in commands_dir.glob("*.md"):
            content = f.read_text()
            content = transform_content(content, "claude")
            # Namespace: plugin_command.md
            dest = target_cmds / f"{plugin_name}_{f.name}"
            dest.write_text(content)
            print(f"    -> Command: {dest.relative_to(root)}")

def main():
    parser = argparse.ArgumentParser(description="Plugin Bridge Installer")
    parser.add_argument("--plugin", required=True, help="Path to plugin directory")
    parser.add_argument("--target", default="auto", choices=["auto", "antigravity", "github", "gemini", "claude"], help="Target environment")
    args = parser.parse_args()

    plugin_path = Path(args.plugin).resolve()
    if not plugin_path.exists():
        print(f"Error: Plugin path not found: {plugin_path}")
        sys.exit(1)

    # Read Metadata
    manifest = plugin_path / ".claude-plugin" / "plugin.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
    else:
        metadata = {"name": plugin_path.name}

    root = Path.cwd()
    targets = []
    
    if args.target == "auto":
        targets = detect_targets(root)
        if not targets:
            print("No compatible environments detected (.agent, .github, .gemini, .claude).")
            sys.exit(1)
    else:
        targets = [args.target]

    print(f"Installing plugin '{metadata['name']}' to: {', '.join(targets)}")

    for t in targets:
        if t == "antigravity":
            install_antigravity(plugin_path, root, metadata)
        elif t == "github":
            install_github(plugin_path, root, metadata)
        elif t == "gemini":
            install_gemini(plugin_path, root, metadata)
        elif t == "claude":
            install_claude(plugin_path, root, metadata)

    print("Installation complete.")

if __name__ == "__main__":
    main()
