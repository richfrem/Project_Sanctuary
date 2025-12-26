#============================================
# mcp_servers/lib/sse_adaptor.py
# Purpose: Gateway-Compatible SSE Transport for MCP Servers
# Status: ACTIVE (Required for Gateway Fleet per ADR-066 v1.3)
# Role: Library/Middleware
# Used by: All Gateway-facing fleet containers (sanctuary_*)
#============================================
# SSEServer: Gateway-Compatible MCP Server Implementation
#
# This is the REQUIRED transport for Gateway-facing fleet containers.
# FastMCP's SSE transport is NOT compatible with the IBM ContextForge Gateway.
#
# See ADR-066 v1.3: "MCP Server Transport Standards (Dual-Stack)"
#
# Pattern (MCP SSE Specification):
#   1. Client GET /sse -> Establishes permanent connection
#   2. Server immediately sends 'endpoint' event with POST URL
#   3. Client POST /messages -> Returns 202 Accepted immediately
#   4. Server processes request -> Pushes JSON-RPC response event to /sse
#   5. Server sends heartbeat pings every 15 seconds
#
# Usage:
#   from mcp_servers.lib.sse_adaptor import SSEServer
#   
#   server = SSEServer("sanctuary_utils")
#   server.register_tool("time.get_current_time", handler, schema)
#   server.run(port=8000, transport="sse")
#============================================

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Awaitable
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# JSON-RPC 2.0 Types
JsonRpcMessage = Dict[str, Any]

# Configurable heartbeat interval (ADR-066 v1.3 recommendation)
HEARTBEAT_INTERVAL = int(os.getenv("SSE_HEARTBEAT_SECONDS", 15))


#============================================
# Decorator: @sse_tool (ADR-076)
# Purpose: Attach metadata to SSE handler functions for Gateway tool discovery.
# Usage:
#   @sse_tool(
#       name="cortex_query",
#       description="Perform semantic search query.",
#       schema=QUERY_SCHEMA
#   )
#   def cortex_query(query: str, max_results: int = 5):
#       ...
#============================================
def sse_tool(
    name: str = None,
    description: str = None,
    schema: dict = None
):
    """
    Decorator to mark functions as SSE tools with explicit metadata.
    
    Per ADR-076, this is the SSE-transport counterpart to FastMCP's @mcp.tool().
    Both decorators delegate to shared operations.py logic.
    
    Args:
        name: Tool name (uses function name if not provided)
        description: Tool description for LLM discovery (uses docstring fallback)
        schema: JSON Schema for input validation
    
    Returns:
        Decorated function with _sse_tool metadata attributes
    """
    def decorator(func):
        func._sse_tool = True
        func._sse_name = name or func.__name__
        func._sse_description = description or (func.__doc__.strip() if func.__doc__ else "No description")
        func._sse_schema = schema or {"type": "object", "properties": {}}
        return func
    return decorator


class SSEServer:
    """
    Gateway-Compatible MCP Server using the Deferred Response Pattern.
    
    Required for IBM ContextForge Gateway integration.
    See ADR-066 v1.3 for transport selection guidance.
    """
    
    #============================================
    # Method: __init__
    # Purpose: Initialize the SSE server with FastAPI routes.
    # Args:
    #   name: Server identification name (e.g., "sanctuary_utils")
    #   version: Semantic version string
    # Returns: None
    #============================================
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.app = FastAPI(title=name, version=version)
        self.logger = logging.getLogger(name)
        
        # Registry: { "tool_name": { "handler": func, "schema": dict } }
        self.tools: Dict[str, Dict[str, Any]] = {}
        
        # Communication queue for SSE message streaming
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        # Setup Routes (MCP SSE Spec)
        self.app.add_api_route("/sse", self.handle_sse, methods=["GET"])
        self.app.add_api_route("/messages", self.handle_messages, methods=["POST"])
        self.app.add_api_route("/health", self.handle_health, methods=["GET"])

    #============================================
    # Method: register_tool
    # Purpose: Register an async tool function with the server.
    # Args:
    #   name: Tool name (should use domain prefix, e.g., "time.get_current_time")
    #   handler: Async function that implements the tool logic
    #   schema: JSON Schema for input validation (optional)
    #   description: Explicit description (optional, falls back to handler docstring)
    # Returns: None
    #============================================
    def register_tool(
        self, 
        name: str, 
        handler: Callable[..., Awaitable[Any]], 
        schema: Optional[Dict] = None,
        description: str = None
    ):
        self.tools[name] = {
            "handler": handler,
            "schema": schema,
            "description": description or (handler.__doc__.strip() if handler.__doc__ else "No description")
        }
        self.logger.info(f"Registered tool: {name}")

    #============================================
    # Method: register_decorated_tools (ADR-076)
    # Purpose: Auto-register all functions decorated with @sse_tool.
    # Args:
    #   namespace: Dict from locals() containing decorated functions
    # Returns: None
    # Note: Implements namespace safety - ignores private functions (starting with _)
    #============================================
    def register_decorated_tools(self, namespace: dict):
        """
        Auto-register all functions decorated with @sse_tool.
        
        Usage:
            server.register_decorated_tools(locals())
        """
        for name, obj in namespace.items():
            # Namespace safety: skip private functions (Red Team hardening)
            if name.startswith('_'):
                continue
            if callable(obj) and getattr(obj, '_sse_tool', False):
                self.register_tool(
                    name=obj._sse_name,
                    handler=obj,
                    schema=obj._sse_schema,
                    description=obj._sse_description
                )
                self.logger.info(f"Auto-registered decorated tool: {obj._sse_name}")

    #============================================
    # Method: handle_sse
    # Purpose: Handle GET /sse - Establish persistent SSE connection with Gateway.
    # Args:
    #   request: FastAPI Request object
    # Returns: EventSourceResponse with message stream
    # Note: This is the critical handshake endpoint. Must immediately send 'endpoint' event.
    #============================================
    async def handle_sse(self, request: Request):
        self.logger.info("SSE Client Connected")
        
        async def event_generator():
            # CRITICAL: Initial handshake - Gateway expects this immediately
            yield {
                "event": "endpoint", 
                "data": "/messages"
            }
            
            # Keep-alive loop with heartbeat pings
            while True:
                if await request.is_disconnected():
                    self.logger.info("SSE Client Disconnected")
                    break
                    
                try:
                    # Wait for message with timeout to allow heartbeat
                    message = await asyncio.wait_for(
                        self._message_queue.get(), 
                        timeout=float(HEARTBEAT_INTERVAL)
                    )
                    
                    self.logger.debug(f"Sending SSE event: {message.get('id', 'unknown')}")
                    yield {
                        "event": "message",
                        "data": json.dumps(message)
                    }
                    self._message_queue.task_done()
                except asyncio.TimeoutError:
                    # Heartbeat ping (keeps connection alive)
                    yield {
                        "event": "ping",
                        "data": "{}"
                    }
                    
        return EventSourceResponse(event_generator())

    #============================================
    # Method: handle_messages
    # Purpose: Handle POST /messages - Receive JSON-RPC requests from Gateway.
    # Args:
    #   request: FastAPI Request object containing JSON-RPC body
    # Returns: HTTP 202 Accepted (immediately, processing is async)
    # Note: Gateway requires immediate 202, response pushed to SSE stream.
    #============================================
    async def handle_messages(self, request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            
        self.logger.info(f"Received JSON-RPC message: {body.get('method')} ID: {body.get('id')}")
        
        # CRITICAL: Return 202 immediately, process async
        asyncio.create_task(self._process_rpc_message(body))
        
        return Response(status_code=202)

    #============================================
    # Method: handle_health
    # Purpose: Health check endpoint for container orchestration.
    # Args: None
    # Returns: JSON {"status": "healthy"}
    #============================================
    async def handle_health(self):
        return {"status": "healthy"}

    #============================================
    # Method: run_stdio
    # Purpose: Run server in STDIO mode for local development.
    # Args: None
    # Returns: None (runs until stdin closes)
    # Note: Alternative to SSE for local debugging with Claude Desktop.
    #============================================
    async def run_stdio(self):
        import sys
        
        stdin = sys.stdin
        self.logger.info("Starting stdio server...")
        
        while True:
            try:
                line = await asyncio.to_thread(stdin.readline)
                if not line:
                    break
                    
                message = json.loads(line)
                response = await self._process_rpc_message_direct(message)
                if response:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Stdio Error: {e}")

    #============================================
    # Method: _process_rpc_message_direct
    # Purpose: Process JSON-RPC message and return response directly.
    # Args:
    #   message: JSON-RPC request dict
    # Returns: JSON-RPC response dict or None for notifications
    # Note: Core RPC handler shared by SSE and STDIO modes.
    #============================================
    async def _process_rpc_message_direct(self, message: JsonRpcMessage) -> Optional[Dict]:
        rpc_id = message.get("id")
        method = message.get("method")
        
        response = {
            "jsonrpc": "2.0", 
            "id": rpc_id
        }
        
        try:
            if method == "tools/list":
                tool_list = []
                for name, info in self.tools.items():
                    schema = info.get("schema") or {"type": "object", "properties": {}}
                    tool_list.append({
                        "name": name,
                        "description": info["description"],
                        "inputSchema": schema
                    })
                response["result"] = {"tools": tool_list}
                
            elif method == "tools/call":
                params = message.get("params", {})
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                # ADR-076: Normalize tool name (Gateway uses hyphens, servers use underscores)
                # e.g. "code-read" -> "code_read"
                normalized_name = tool_name.replace("-", "_") if tool_name else tool_name
                
                # Try normalized name first, then original
                if normalized_name in self.tools:
                    handler = self.tools[normalized_name]["handler"]
                elif tool_name in self.tools:
                    handler = self.tools[tool_name]["handler"]
                else:
                    raise Exception(f"Tool not found: {tool_name} (tried: {normalized_name})")
                
                # Call handler (async or sync)
                if asyncio.iscoroutinefunction(handler):
                    result_content = await handler(**tool_args)
                else:
                    result_content = handler(**tool_args)
                
                content_list = []
                if isinstance(result_content, str):
                    content_list.append({"type": "text", "text": result_content})
                elif isinstance(result_content, (dict, list)):
                    content_list.append({"type": "text", "text": json.dumps(result_content, indent=2)})
                else:
                    content_list.append({"type": "text", "text": str(result_content)})
                    
                response["result"] = {"content": content_list}
            
            elif method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.name, "version": self.version},
                    "capabilities": {"tools": {}}
                }
            
            elif method == "notifications/initialized":
                return None
                
            else:
                # Handle ping and unknown methods gracefully
                if method and method.startswith("ping"):
                    return None
                raise Exception(f"Method not found: {method}")

        except Exception as e:
            self.logger.exception(f"RPC Error: {method}")
            response["error"] = {"code": -32603, "message": str(e)}
            if "result" in response:
                del response["result"]
            
        return response

    #============================================
    # Method: _process_rpc_message
    # Purpose: SSE wrapper - process message and queue response for streaming.
    # Args:
    #   message: JSON-RPC request dict
    # Returns: None (queues response to SSE stream)
    #============================================
    async def _process_rpc_message(self, message: JsonRpcMessage):
        response = await self._process_rpc_message_direct(message)
        if response:
            await self._message_queue.put(response)

    #============================================
    # Method: run
    # Purpose: Start the server in configured transport mode.
    # Args:
    #   port: Port for SSE/HTTP mode (default 8000)
    #   transport: 'sse' or 'stdio' (can be overridden by MCP_TRANSPORT env)
    # Returns: None (blocks until shutdown)
    # Note: See ADR-066 v1.3 for transport selection guidance.
    #============================================
    def run(self, port: int = 8000, transport: str = "sse"):
        env_transport = os.getenv("MCP_TRANSPORT", transport).lower()
        
        if env_transport == "stdio":
            asyncio.run(self.run_stdio())
        else:
            import uvicorn
            env_port = int(os.getenv("PORT", port))
            
            self.logger.info(f"Starting SSE server on port {env_port}")
            uvicorn.run(self.app, host="0.0.0.0", port=env_port, log_level="info")
