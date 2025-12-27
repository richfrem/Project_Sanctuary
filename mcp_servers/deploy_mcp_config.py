#!/usr/bin/env python3
#============================================
# mcp_servers/deploy_mcp_config.py
# Purpose: Deploy MCP config from env-template (cross-platform).
# Role: Deployment Script
#============================================
"""
Deploy MCP config from env-template (cross-platform)
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Import Utilities
project_root_search = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(project_root_search) not in sys.path:
    sys.path.append(os.path.dirname(project_root_search))

from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.logging_utils import setup_mcp_logging

# Setup Logging (optional for CLI, but good for consistency/debug)
logger = setup_mcp_logging("deploy_mcp_config")


def get_defaults() -> dict:
    script_dir = Path(__file__).resolve().parent
    project_root = (script_dir / "..").resolve()
    template = project_root / ".agent" / "mcp_config.json"
    return {"script_dir": script_dir, "project_root": project_root, "template": template}


def resolve_dest(target: str, project_root: Path) -> Path:
    sysplt = platform.system()
    home = Path.home()
    t = target.lower()
    if t == "claudedesktop":
        if sysplt == "Darwin":
            default = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
            if default.exists():
                return default
            # Fallback: try to discover any existing Claude config under the home directory
            for p in home.rglob('claude*config*.json'):
                # prefer exact name first
                if p.name == 'claude_desktop_config.json':
                    return p
            # otherwise return the default path (will be created)
            return default
        elif sysplt == "Windows":
            appdata = get_env_variable("APPDATA", default="%APPDATA%")
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        else:
            appdata = get_env_variable("XDG_CONFIG_HOME", default=str(home / ".config"))
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    if t == "antigravity":
        if sysplt in ("Darwin", "Linux"):
            return home / ".gemini" / "antigravity" / "mcp_config.json"
        else:
            appdata = get_env_variable("APPDATA", default="%APPDATA%")
            return Path(appdata) / "Gemini" / "Antigravity" / "mcp_config.json"
    if t == "relativemcp":
        return home / "mcp"
    if t == "vscodeworkspace":
        return project_root / ".vscode" / "mcp.json"
    if t in ("vscodeuser", "vscodetuser"):
        if sysplt == "Darwin":
            return home / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
        elif sysplt == "Windows":
            appdata = get_env_variable("APPDATA", default=str(home / "AppData" / "Roaming"))
            return Path(appdata) / "Code" / "User" / "mcp.json"
        else:
            return home / ".config" / "Code" / "User" / "mcp.json"
    raise ValueError(f"Unknown target: {target}")


def expand_template(template_path: Path) -> str:
    s = template_path.read_text(encoding="utf8")
    return os.path.expandvars(s)


def backup_file(path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_name(path.name + f".{ts}.bak")
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, bak)
    return bak


def write_file(path: Path, content: str, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] Would write {path}")
        return
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(content, encoding="utf8")
    os.replace(tmp, path)
    if platform.system() in ("Darwin", "Linux"):
        try:
            path.chmod(0o600)
        except Exception:
            pass


def merge_vscode_settings(settings_path: Path, mcp_config_path_value: str, dry_run: bool) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf8"))
        except Exception:
            data = {}
    data["mcp.configPath"] = mcp_config_path_value
    content = json.dumps(data, indent=2, ensure_ascii=False)
    if dry_run:
        print(f"[dry-run] Would update VS Code settings at {settings_path} to set mcp.configPath = {mcp_config_path_value}")
        return
    tmp = settings_path.with_name(settings_path.name + ".new")
    tmp.write_text(content, encoding="utf8")
    os.replace(tmp, settings_path)


def main(argv: list[str] | None = None) -> int:
    defaults = get_defaults()
    parser = argparse.ArgumentParser(description="Deploy MCP config from env-template into platform config locations")
    parser.add_argument("--target", required=True, help="Target to update (ClaudeDesktop, Antigravity, RelativeMCP, VSCodeWorkspace, VSCodeUser)")
    parser.add_argument("--template", default=str(defaults["template"]), help="Path to env-template JSON")
    parser.add_argument("--backup", action="store_true", help="Backup existing destination file")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files; just show actions")
    parser.add_argument("--preserve-placeholders", action="store_true", help="Keep ${...} placeholders for known vars instead of absolute expansion")
    parser.add_argument("--add-legacy-project-root", action="store_true", help="Also add PROJECT_ROOT entries mirroring PROJECT_SANCTUARY_ROOT for backward compatibility")
    parser.add_argument("--project-root", help="Override project root path (defaults to repository parent of this script)")

    args = parser.parse_args(argv)
    project_root = Path(args.project_root) if args.project_root else defaults["project_root"]
    template_path = Path(args.template).expanduser().resolve()
    if not template_path.exists():
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 2

    try:
        dest = resolve_dest(args.target, project_root)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 3

    print(f"Template: {template_path}")
    print(f"Destination: {dest}")
    # Expand template early so preview and validation can use it
    expanded = expand_template(template_path)

    # Optionally reverse some expansions to keep placeholders for downstream clients
    if args.preserve_placeholders:
        # Replace absolute project root occurrences with the placeholder
        pr = str(project_root)
        expanded = expanded.replace(pr, '${PROJECT_SANCTUARY_ROOT}')
        # Replace known virtualenv python path with placeholder if present
        pyenv = get_env_variable('PYTHON_EXEC', required=False)
        if pyenv:
            expanded = expanded.replace(pyenv, '${PYTHON_EXEC}')
        else:
            # common venv location fallback
            fallback_py = str(project_root / '.venv' / 'bin' / 'python')
            expanded = expanded.replace(fallback_py, '${PYTHON_EXEC}')

    # Optionally add legacy PROJECT_ROOT alias into the JSON env blocks
    if args.add_legacy_project_root:
        try:
            data = json.loads(expanded)
            if isinstance(data, dict) and 'mcpServers' in data and isinstance(data['mcpServers'], dict):
                for name, entry in data['mcpServers'].items():
                    if isinstance(entry, dict) and 'env' in entry and isinstance(entry['env'], dict):
                        # prefer PROJECT_SANCTUARY_ROOT if present, else use cwd or existing PROJECT_ROOT
                        if 'PROJECT_SANCTUARY_ROOT' in entry['env']:
                            entry['env']['PROJECT_ROOT'] = entry['env']['PROJECT_SANCTUARY_ROOT']
                        elif 'PROJECT_ROOT' not in entry['env']:
                            entry['env']['PROJECT_ROOT'] = '${PROJECT_SANCTUARY_ROOT}'
                expanded = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            # if parsing fails, leave expanded as-is
            pass

    if args.dry_run:
        print("Dry run: no changes will be written")
        # show a preview of the expanded template to help debugging
        preview = expanded if len(expanded) <= 8000 else expanded[:8000] + "\n... (truncated)"
        print("\n--- Expanded template preview ---\n")
        print(preview)
        print("\n--- end preview ---\n")

    if args.backup and dest.exists():
        bak = backup_file(dest)
        print(f"Backing up existing file to: {bak}")

    if not args.dry_run:
        print(f"Writing expanded config to: {dest}")
    write_file(dest, expanded, args.dry_run)
    print(f"Updated: {dest}")

    t = args.target.lower()
    if t == "vscodeworkspace":
        settings = project_root / ".vscode" / "settings.json"
        value = '${workspaceFolder}/.vscode/mcp.json'
        merge_vscode_settings(settings, value, args.dry_run)
        print(f"Updated workspace settings: {settings} -> mcp.configPath = {value}")
    elif t in ("vscodeuser", "vscodetuser"):
        sysplt = platform.system()
        home = Path.home()
        if sysplt == "Darwin":
            settings = home / "Library" / "Application Support" / "Code" / "User" / "settings.json"
            mcp_user = home / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
        elif sysplt == "Windows":
            appdata = get_env_variable("APPDATA", default=str(home / "AppData" / "Roaming"))
            settings = Path(appdata) / "Code" / "User" / "settings.json"
            mcp_user = Path(appdata) / "Code" / "User" / "mcp.json"
        else:
            settings = home / ".config" / "Code" / "User" / "settings.json"
            mcp_user = home / ".config" / "Code" / "User" / "mcp.json"
        merge_vscode_settings(settings, str(mcp_user), args.dry_run)
        print(f"Updated user settings: {settings} -> mcp.configPath = {mcp_user}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
