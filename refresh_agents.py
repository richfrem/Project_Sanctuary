#!/usr/bin/env python3
"""
Refresh Agent Environment
=========================

Force refreshes the agent environment by:
1. Re-running install_all_plugins.py to update agent directories.
2. Synchronizing skills and workflows via spec-kitty bridge.
3. Refreshing the RLM cache (optional, but good for hygiene).

Usage:
    python3 refresh_agents.py
"""

import sys
import subprocess
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
PLUGINS_DIR = PROJECT_ROOT / "plugins"

def run_command(cmd, cwd=None):
    print(f"🚀 Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        print("   ✅ Success")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    print("🔄 Refreshing Agent Environment...\n")
    
    # 1. Install Plugins (Updates .agent/skills and others)
    install_script = PLUGINS_DIR / "plugin-manager" / "skills" / "plugin-manager" / "scripts" / "install_all_plugins.py"
    if install_script.exists():
        run_command([sys.executable, str(install_script)])
    else:
        print(f"❌ Could not find install script: {install_script}")
        sys.exit(1)

    # 2. Spec-Kitty Sync (Bridge)
    # We can rely on install_all_plugins.py doing some of this, but let's be explicit if needed.
    # Actually, install_all_plugins.py calls bridge_installer.py which does the copy.
    # The 'Bridge' (speckit_system_bridge.py) is for syncing workflows from .windsurf -> everyone.
    
    bridge_script = PLUGINS_DIR / "spec-kitty" / "skills" / "spec-kitty-agent" / "scripts" / "speckit_system_bridge.py"
    if bridge_script.exists():
        print("\n🌉 Running Universal Bridge Sync...")
        run_command([sys.executable, str(bridge_script)])

    print("\n✨ Agent Environment Refreshed!")

if __name__ == "__main__":
    main()
