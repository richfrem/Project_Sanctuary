#!/usr/bin/env python3
"""
MCP Gateway Bridge - Stdio to SSE/HTTP
Bridges Claude Desktop (stdio) to Sanctuary Gateway (SSE/HTTP)
Based on ADR 068 - Option B: Single File Approach
"""
import asyncio
import json
import os
import sys
from typing import Optional
import logging # Added correct import

try:
    import httpx
    # Use project standard helper
    from mcp_servers.lib.env_helper import get_env_variable
except ImportError:
    # If dotenv is missing, we proceed; token must be in env vars
    pass
except Exception:
    pass

if "httpx" not in sys.modules:
    try:
        import httpx
    except ImportError:
        print(json.dumps({"error": "httpx not installed. Run: pip install httpx"}), file=sys.stderr)
        sys.exit(1)


class GatewayBridge:
    def __init__(self):
        # Use helper used across project for consistency
        self.gateway_url = get_env_variable("MCP_GATEWAY_URL", required=False) or "https://localhost:4444"
        self.token = get_env_variable("MCPGATEWAY_BEARER_TOKEN", required=True)
        
        ssl_val = get_env_variable("MCP_GATEWAY_VERIFY_SSL", required=False)
        self.verify_ssl = str(ssl_val).lower() != "false" if ssl_val else False
        
        # Setup debug logging to file
        logging.basicConfig(
            filename='/tmp/bridge_debug.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if not self.token:
            raise ValueError("MCPGATEWAY_BEARER_TOKEN environment variable is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        self.client = httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=30.0,
            follow_redirects=True
        )
    
    async def handle_request(self, request: dict) -> Optional[dict]:
        """Forward MCP request to gateway via HTTP"""
        self.logger.debug(f"Received Request: {json.dumps(request)}")
        method = request.get("method", "")
        request_id = request.get("id")
        
        # Handle notifications (no ID) - specifically initialized
        if request_id is None:
            self.logger.debug(f"Ignoring notification: {method}")
            return None

        # Map MCP methods to Gateway HTTP endpoints
        if method == "initialize":
            return await self.initialize(request_id)
        elif method == "tools/list":
            return await self.list_tools(request_id)
        elif method == "tools/call":
            return await self.call_tool(request)
        elif method == "resources/list":
            return await self.list_resources(request_id)
        elif method == "prompts/list":
            return await self.list_prompts(request_id)
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": request_id
            }
    
    async def initialize(self, request_id) -> dict:
        """Initialize connection with gateway"""
        try:
            response = await self.client.get(
                f"{self.gateway_url}/health",
                headers=self.headers
            )
            response.raise_for_status()
            
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "sanctuary-gateway",
                        "version": "1.0.0"
                    }
                },
                "id": request_id
            }
        except Exception as e:
            return self.error_response(f"Failed to initialize: {e}", request_id)
    
    async def list_tools(self, request_id) -> dict:
        """List all available tools from gateway (handling pagination)"""
        try:
            all_tools = []
            cursor = None
            
            while True:
                params = {"per_page": 200}
                if cursor:
                    params["cursor"] = cursor
                
                # Use /admin/tools endpoint like gateway_client.py to bypass 50 limit
                response = await self.client.get(
                    f"{self.gateway_url}/admin/tools",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                # Handle structure
                # MCP JSON-RPC response format for tools/list usually has { result: { tools: [], nextCursor: ... } }
                # But Gateway HTTP /tools endpoint might return the inner result directly or a wrapper.
                # Assuming typical MCP over HTTP or the structure observed in logs (+ cursor support)
                
                # Check if 'data' is the list itself or a wrapper
                if isinstance(data, list):
                    # If it's just a list, assume no pagination or legacy format
                    all_tools.extend(data)
                    break
                
                # If dict, look for 'tools' or 'data' and 'nextCursor'
                # Standard MCP list_tools result: { "tools": [...], "nextCursor": "..." }
                # Or Gateway specific wrapper: { "data": [...], "next_cursor": ... }
                
                batch = []
                next_cursor = None
                
                if "tools" in data:
                    batch = data["tools"]
                    next_cursor = data.get("nextCursor")
                elif "data" in data:
                    batch = data["data"]
                    next_cursor = data.get("nextCursor") or data.get("next_cursor")
                elif "result" in data: # Nested JSON-RPC-like structure
                    result = data["result"]
                    batch = result.get("tools", [])
                    next_cursor = result.get("nextCursor")
                else:
                    # Fallback
                    batch = []

                all_tools.extend(batch)
                
                if not next_cursor:
                    break
                    
                cursor = next_cursor

            # Convert gateway tool format to MCP format
            mcp_tools = []
            for tool in all_tools:
                original_name = tool.get("name")
                # Strip 'sanctuary-' prefix to shorten name for client validation
                short_name = original_name.replace("sanctuary-", "", 1) if original_name.startswith("sanctuary-") else original_name
                
                mcp_tools.append({
                    "name": short_name,
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {"type": "object", "properties": {}})
                })
            
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": mcp_tools
                },
                "id": request_id
            }
        except Exception as e:
            self.logger.error(f"List tools error: {e}")
            return self.error_response(f"Failed to list tools: {e}", request_id)
    
    async def call_tool(self, request: dict) -> dict:
        """Call a tool via gateway"""
        try:
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Re-add 'sanctuary-' prefix if strictly mapped, or just pass through
            # We assume if it doesn't start with sanctuary-, we added it back
            real_tool_name = tool_name
            if not tool_name.startswith("sanctuary-"):
                real_tool_name = f"sanctuary-{tool_name}"

            # Call tool via gateway JSON-RPC endpoint
            rpc_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": real_tool_name,
                    "arguments": arguments
                },
                "id": request.get("id", 1)
            }
            
            response = await self.client.post(
                f"{self.gateway_url}/rpc",
                headers=self.headers,
                json=rpc_request
            )
            response.raise_for_status()
            
            gateway_response = response.json()
            
            # Convert gateway response to MCP format
            if "result" in gateway_response:
                result = gateway_response["result"]
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": result.get("content", []),
                        "isError": result.get("isError", False)
                    },
                    "id": request.get("id")
                }
            elif "error" in gateway_response:
                return gateway_response
            else:
                return self.error_response("Invalid gateway response", request.get("id"))
                
        except Exception as e:
            return self.error_response(f"Failed to call tool: {e}", request.get("id"))
    
    async def list_resources(self, request_id) -> dict:
        """List resources (placeholder)"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "resources": []
            },
            "id": request_id
        }
    
    async def list_prompts(self, request_id) -> dict:
        """List prompts (placeholder)"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "prompts": []
            },
            "id": request_id
        }
    
    def error_response(self, message: str, request_id: Optional[int] = None) -> dict:
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": message
            },
            "id": request_id
        }
    
    async def run(self):
        """Main loop: read from stdin, forward to gateway, write to stdout"""
        try:
            while True:
                # Read JSON-RPC request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    response = await self.handle_request(request)
                    if response:
                        print(json.dumps(response), flush=True)
                    
                except json.JSONDecodeError as e:
                    error = self.error_response(f"Parse error: {e}")
                    print(json.dumps(error), flush=True)
                    
        except KeyboardInterrupt:
            pass
        finally:
            await self.client.aclose()


async def main():
    try:
        bridge = GatewayBridge()
        await bridge.run()
    except Exception as e:
        error = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Bridge initialization failed: {e}"
            },
            "id": None
        }
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
