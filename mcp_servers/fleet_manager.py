import subprocess
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Re-use config from start_mcp_servers
from mcp_servers.start_mcp_servers import MODULES_TO_START, PROJECT_ROOT, PY_EXECUTABLE

class MCPServerFleet:
    """
    Manages the lifecycle of the entire MCP server fleet for E2E testing.
    """
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logs: Dict[str, Path] = {}
        self.log_dir = PROJECT_ROOT / "logs" / "e2e_fleet"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start all servers in background, capturing output."""
        print(f"Booting MCP Fleet ({len(MODULES_TO_START)} servers)...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        for mod in MODULES_TO_START:
            server_name = mod.split(".")[-2] # mcp_servers.adr.server -> adr
            
            # Create log files
            stdout_path = self.log_dir / f"{server_name}.stdout.log"
            stderr_path = self.log_dir / f"{server_name}.stderr.log"
            
            stdout_f = open(stdout_path, "w")
            stderr_f = open(stderr_path, "w")
            
            cmd = [PY_EXECUTABLE, "-m", mod]
            
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=PROJECT_ROOT,
                env=env,
                text=True
            )
            self.processes[server_name] = proc
            print(f"  Started {server_name} (PID {proc.pid})")
            
        # Wait a moment for startup
        time.sleep(2)
        print("Fleet operational.")
        
    def stop(self):
        """Stop all managed servers."""
        print("Stopping MCP Fleet...")
        for name, proc in self.processes.items():
            if proc.poll() is None:
                proc.terminate()
        
        # Wait for termination
        time.sleep(1)
        for name, proc in self.processes.items():
            if proc.poll() is None:
                print(f"  Killing stuck server: {name}")
                proc.kill()
        print("Fleet shutdown.")

    def get_server_pid(self, name: str) -> int:
        if name in self.processes:
            return self.processes[name].pid
        return -1
