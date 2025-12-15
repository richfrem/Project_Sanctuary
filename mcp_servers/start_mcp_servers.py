#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Project Sanctuary MCP Server Launcher for VS Code (GitHub Copilot)
# ------------------------------------------------------------------------------
from __future__ import annotations

import subprocess
import sys
import argparse
import time
import os
from pathlib import Path

# --- Configuration (portable, resolved relative to this script) ---
# SCRIPT_DIR is Project_Sanctuary/mcp_servers/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PY_EXECUTABLE = str(PROJECT_ROOT / ".venv" / "bin" / "python")

# Canonical list of all 12 MCP servers
MODULES_TO_START = [
    "mcp_servers.adr.server",
    "mcp_servers.agent_persona.server",
    "mcp_servers.chronicle.server",
    "mcp_servers.code.server",
    "mcp_servers.config.server",
    "mcp_servers.council.server",
    "mcp_servers.forge_llm.server",
    "mcp_servers.git.server",
    "mcp_servers.orchestrator.server",
    "mcp_servers.protocol.server",
    "mcp_servers.rag_cortex.server",
    "mcp_servers.task.server",
]

def launch_background_servers():
    """Launches all servers in the background for VS Code auto-start."""
    print(f"Starting all 12 Project Sanctuary MCP Servers from {PROJECT_ROOT}...")
    
    # Change directory to project root for module path resolution
    os.chdir(PROJECT_ROOT)

    for mod in MODULES_TO_START:
        try:
            # Use Popen for detached background execution
            # Redirecting to PIPE prevents stdout pollution of the primary process
            subprocess.Popen(
                [PY_EXECUTABLE, "-m", mod], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=PROJECT_ROOT
            )
            print(f"  ✅ Launched: {mod}")
            time.sleep(0.1) 
        except Exception as e:
            print(f"  ❌ Failed to launch {mod}: {e}", file=sys.stderr)

    print("---\nAll background launch commands issued. The Task will now exit successfully.")

def get_server_file_paths() -> list[Path]:
    """Converts module names to actual file paths for existence verification."""
    paths = []
    for mod in MODULES_TO_START:
        # Convert 'mcp_servers.adr.server' -> 'mcp_servers/adr/server.py'
        rel_path = mod.replace(".", "/") + ".py"
        paths.append(PROJECT_ROOT / rel_path)
    return paths

def main():
    parser = argparse.ArgumentParser(description="Start Project Sanctuary MCP servers")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    parser.add_argument("--run", action="store_true", help="Launch servers in the foreground (wait for exit)")
    args = parser.parse_args()

    # 1. Verify files exist before attempting any launch
    paths = get_server_file_paths()
    missing = [str(p) for p in paths if not p.exists()]
    
    if missing:
        for m in missing:
            print(f"ERROR: Server file not found: {m}", file=sys.stderr)
        sys.exit(2)

    # 2. Mode: Dry Run (Just print commands)
    if args.dry_run:
        print(f"Using Python interpreter: {PY_EXECUTABLE}")
        print("Server run commands (Dry-run):")
        for p in paths:
            print(f"  {PY_EXECUTABLE} {p}")
        return

    # 3. Mode: Foreground (Wait for servers, useful for debugging)
    if args.run:
        procs = []
        try:
            os.chdir(PROJECT_ROOT) # Ensure cwd is root for module imports
            for mod in MODULES_TO_START:
                print(f"Launching Foreground: {PY_EXECUTABLE} -m {mod}")
                procs.append(subprocess.Popen([PY_EXECUTABLE, "-m", mod], cwd=PROJECT_ROOT))
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            print("\nShutting down servers...")
            for p in procs: 
                p.terminate()
        return

    # 4. Mode: Default (Used by VS Code Task)
    # Launches background processes and exits, letting the Task finish.
    launch_background_servers()

if __name__ == "__main__":
    main()