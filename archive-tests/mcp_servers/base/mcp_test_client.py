import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPTestClient:
    """
    Test Client for Headless MCP Servers.
    
    A lightweight MCP Client for E2E testing.
    Spawns a server process and communicates via JSON-RPC 2.0 over stdio.
    """
    def __init__(self, server_target: Union[Path, str], is_module: bool = False):
        self.server_target = server_target
        self.is_module = is_module
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self._notification_handlers = {}
        
        # Determine name for logging
        if self.is_module:
            self.name = str(server_target).split(".")[-2] if "." in str(server_target) else str(server_target)
        else:
            self.name = Path(server_target).name

    def start(self, env: Optional[Dict[str, str]] = None):
        """Start the MCP server subprocess."""
        # Use sys.executable to ensure we use the same Python environment
        if self.is_module:
             cmd = [sys.executable, "-m", str(self.server_target)]
        else:
             cmd = [sys.executable, str(self.server_target)]
             
        logger.info(f"Starting MCP Server: {' '.join(cmd)}")
        
        # Line buffering usually requires env var or -u, but bufsize=1 + text=True helps in Python
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout to capture logs
            text=True,
            bufsize=1, # Line buffered
            env=env or os.environ.copy()
        )
        logger.info(f"Server started (PID {self.process.pid})")
        
        # Perform Initialization Handshake
        try:
            self.initialize()
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to initialize server: {e}")

    def initialize(self):
        """Perform the MCP initialization handshake."""
        self.request_id += 1
        init_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", # Or "latest"
                "capabilities": {
                    "sampling": {},
                    "roots": {"listChanged": True}
                },
                "clientInfo": {"name": "MCPTestClient", "version": "1.0.0"}
            }
        }
        
        # Send Initialize
        json_str = json.dumps(init_request) + "\n"
        self.process.stdin.write(json_str)
        self.process.stdin.flush()
        
        # Read Response
        resp = self._read_response()
        if resp.get("id") != self.request_id:
            raise RuntimeError(f"Unexpected response ID during init: {resp}")
        
        if "error" in resp:
            raise RuntimeError(f"Initialization failed: {resp['error']}")
            
        logger.info(f"Server initialized. Capabilities: {resp.get('result', {}).get('capabilities')}")
        
        # Send Initialized Notification
        notify = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        notify_str = json.dumps(notify) + "\n"
        self.process.stdin.write(notify_str)
        self.process.stdin.flush()
        # No response expected for notification
        
    def stop(self):
        """Stop the server subprocess."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Server stopped")
            
    def _read_response(self, timeout: float = 60.0) -> Dict[str, Any]:
        """Blocking read of the next JSON-RPC message from stdout with timeout.
        
        Args:
            timeout: Maximum seconds to wait for response (default: 60s)
        """
        import select
        
        if not self.process or not self.process.stdout:
            raise RuntimeError("Server not running")
        
        start_time = time.time()
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Server response timeout after {timeout}s")
            
            # Use select to check if data is available (with remaining timeout)
            remaining = timeout - elapsed
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 1.0))
            
            if not ready:
                # No data available yet, continue loop (will check timeout on next iteration)
                continue
            
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise EOFError(f"Server closed connection. Stderr: {stderr}")
            
            # Skip debug lines if any (hacky, but FastMCP might log)
            if not line.strip().startswith("{"):
                logger.info(f"[{self.name}] Output: {line.strip()}")
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
