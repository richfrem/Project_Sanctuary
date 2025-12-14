import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPTestClient:
    """
    A lightweight MCP Client for E2E testing.
    Spawns a server process and communicates via JSON-RPC 2.0 over stdio.
    """
    def __init__(self, server_path: Path):
        self.server_path = server_path
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self._notification_handlers = {}
        
    def start(self, env: Optional[Dict[str, str]] = None):
        """Start the MCP server subprocess."""
        cmd = ["python", str(self.server_path)]
        logger.info(f"Starting MCP Server: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr to prevent noise
            text=True,
            bufsize=1, # Line buffered
            env=env or os.environ.copy()
        )
        logger.info(f"Server started (PID {self.process.pid})")
        
    def stop(self):
        """Stop the server subprocess."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Server stopped")
            
    def _read_response(self) -> Dict[str, Any]:
        """Blocking read of the next JSON-RPC message from stdout."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Server not running")
            
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise EOFError(f"Server closed connection. Stderr: {stderr}")
            
            # Skip debug lines if any (hacky, but FastMCP might log)
            if not line.strip().startswith("{"):
                # logger.debug(f"Server Output: {line.strip()}")
                continue
                
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from server: {line}")
                continue

    def call_tool(self, name: str, arguments: Dict[str, Any] = {}) -> Any:
        """Call an MCP tool and return the result."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        # Send Request
        if not self.process or not self.process.stdin:
            raise RuntimeError("Server not running")
            
        json_str = json.dumps(request) + "\n"
        self.process.stdin.write(json_str)
        self.process.stdin.flush()
        
        # Read Response
        # Note: In a real protocol, we might receive notifications or other requests.
        # This simple client assumes request-response for now.
        while True:
            response = self._read_response()
            
            if response.get("id") == self.request_id:
                if "error" in response:
                    raise RuntimeError(f"Tool execution failed: {response['error']}")
                return response.get("result", {})
            
            # Handle notifications or requests (e.g., logging)
            method = response.get("method")
            if method == "notifications/message":
                # Log level handling
                pass
            elif method == "logging/message":
                pass
            
            # If it's a request from the server (e.g. sampling), we fail for now (mocking needed)
            if "method" in response and "id" in response:
                 # It's a request FROM server (e.g. sampling)
                 # We must reply to keyalive/ping/etc
                 pass

    def list_tools(self) -> List[Dict[str, Any]]:
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/list",
        }
        json_str = json.dumps(request) + "\n"
        self.process.stdin.write(json_str)
        self.process.stdin.flush()
        
        while True:
            response = self._read_response()
            if response.get("id") == self.request_id:
                return response.get("result", {}).get("tools", [])

