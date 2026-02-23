#!/usr/bin/env python3
"""
Bulk Plugin Installer
=====================

Iterates through all directories in `plugins/` and runs the `bridge_installer.py`
for each one. This ensures a clean, full installation of all available plugins.

Usage:
    python3 plugins/plugin-manager/scripts/install_all_plugins.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent
PLUGINS_ROOT = PROJECT_ROOT / "plugins"

INSTALLER_SCRIPT = SCRIPT_DIR / "bridge_installer.py"

def main():
    if not INSTALLER_SCRIPT.exists():
        print(f"❌ Error: Installer script not found at {INSTALLER_SCRIPT}")
        sys.exit(1)

    print(f"🚀 Starting Batch Installation from {PLUGINS_ROOT}...")
    
    plugins_processed = 0
    plugins_failed = 0
    
    # Iterate over all directories in plugins/
    for plugin_dir in sorted(PLUGINS_ROOT.iterdir()):
        if not plugin_dir.is_dir():
            continue
            
        # Skip special directories
        if plugin_dir.name.startswith(".") or plugin_dir.name.startswith("__"):
            continue
        if plugin_dir.name in ["node_modules", "venv", "env"]:
            continue
            
        print(f"\n📦 Installing: {plugin_dir.name}")
        
        try:
            # Run the bridge installer for this plugin
            # We use subprocess to isolate execution and ensure clean state
            cmd = [
                sys.executable, 
                str(INSTALLER_SCRIPT),
                "--plugin", str(plugin_dir),
                "--target", "auto"
            ]
            
            result = subprocess.run(cmd, check=True, text=True)
            plugins_processed += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {plugin_dir.name}")
            plugins_failed += 1
        except Exception as e:
            print(f"❌ Unexpected error installing {plugin_dir.name}: {e}")
            plugins_failed += 1

    print("\n" + "="*50)
    print(f"Batch Installation Complete")
    print(f"✅ Success: {plugins_processed}")
    if plugins_failed > 0:
        print(f"❌ Failed:  {plugins_failed}")
    print("="*50)

if __name__ == "__main__":
    main()
