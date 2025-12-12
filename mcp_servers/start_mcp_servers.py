#!/usr/bin/env python3
"""
Cross-platform starter for Project Sanctuary MCP servers.

Usage:
  python3 mcp_servers/start_mcp_servers.py [--dry-run] [--run]

By default this verifies server files and prints the exact commands to run them.
Use `--run` to actually spawn the server processes (keeps them running in foreground).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def get_project_root() -> Path:
    # script is located under mcp_servers/; project root is parent
    return Path(__file__).resolve().parent


def default_python_exec() -> str:
    # Prefer env var PYTHON_EXEC if set (launch scripts set this), otherwise use current interpreter
    return os.environ.get("PYTHON_EXEC", sys.executable)


def server_paths(project_root: Path) -> List[Path]:
    return [
        project_root / "cognitive" / "cortex" / "server.py",
        project_root / "chronicle" / "server.py",
        project_root / "protocol" / "server.py",
        project_root / "orchestrator" / "server.py",
    ]


def check_files(paths: List[Path]) -> bool:
    ok = True
    for p in paths:
        if not p.exists():
            print(f"ERROR: Server file not found: {p}")
            ok = False
    return ok


def print_commands(python_exec: str, paths: List[Path]) -> None:
    print("Server run commands:")
    for p in paths:
        print(f"  {python_exec} {p}")


def run_servers(python_exec: str, paths: List[Path]) -> int:
    procs = []
    try:
        for p in paths:
            print(f"Launching: {python_exec} {p}")
            proc = subprocess.Popen([python_exec, str(p)])
            procs.append((p, proc))
        print("Launched processes:")
        for p, proc in procs:
            print(f"  {p} -> PID {proc.pid}")
        # Wait for processes; if any exits, return non-zero
        exit_codes = [proc.wait() for (_, proc) in procs]
        # if all zero -> success
        return 0 if all(code == 0 for code in exit_codes) else 1
    except KeyboardInterrupt:
        print("Interrupted, terminating children...")
        for _, proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Project Sanctuary MCP servers (cross-platform)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    parser.add_argument("--run", action="store_true", help="Actually launch the server processes")
    args = parser.parse_args(argv)

    project_root = get_project_root()
    paths = server_paths(project_root)
    python_exec = default_python_exec()

    if not check_files(paths):
        print("One or more server files are missing. Aborting.")
        return 2

    print(f"Using Python interpreter: {python_exec}")
    print_commands(python_exec, paths)

    if args.dry_run or not args.run:
        print("Dry-run mode, no processes started.")
        return 0

    # Run the servers
    return run_servers(python_exec, paths)


if __name__ == "__main__":
    raise SystemExit(main())
