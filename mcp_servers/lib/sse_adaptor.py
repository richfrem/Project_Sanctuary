
import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Awaitable
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# JSON-RPC 2.0 Types
JsonRpcMessage = Dict[str, Any]

class SSEServer:
    """
    A specific MCP Server implementation that enforces the "Deferred Response" pattern.
    
    Pattern:
    1. Client GET /sse -> Establishes permanent connection.
    2. Client POST /messages -> Returns 202 Accepted immediately.
    3. Server processes request -> Pushes JSON-RPC response event to /sse.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.app = FastAPI(title=name, version=version)
        self.logger = logging.getLogger(name)
        
        # Registry
        # Format: { "tool_name": { "handler": func, "schema": dict } }
        self.tools: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        # Setup Routes
        self.app.add_api_route("/sse", self.handle_sse, methods=["GET"])
        self.app.add_api_route("/messages", self.handle_messages, methods=["POST"])
        self.app.add_api_route("/health", self.handle_health, methods=["GET"])

    def register_tool(self, name: str, handler: Callable[..., Awaitable[Any]], schema: Optional[Dict] = None):
        """Register an async tool function."""
        self.tools[name] = {
            "handler": handler,
            "schema": schema,
            "description": handler.__doc__.strip() if handler.__doc__ else "No description"
        }
        self.logger.info(f"Registered tool: {name}")

    async def handle_sse(self, request: Request):
        """Streams messages to the client."""
        self.logger.info("SSE Client Connected")
        
        async def event_generator():
             # Initial handshake
            yield {
                "event": "endpoint", 
                "data": "/messages" # MCP Spec suggestion or custom
            }
            
            # Keep alive loop
            while True:
                if await request.is_disconnected():
                    self.logger.info("SSE Client Disconnected")
                    break
                    
                try:
                    # Wait for message with short timeout to allow heartbeat
                    # Check for token in header (User feedback: MCPGATEWAY_BEARER_TOKEN)
                    # We don't strictly enforce it here yet as Gateway might handle auth termination,
                    # but we log if it's missing just in case debugging is needed.
                    if "Authorization" not in request.headers and "authorization" not in request.headers:
                        pass # self.logger.debug("No Authorization header in SSE request")

                    message = await asyncio.wait_for(self._message_queue.get(), timeout=15.0)
                    
                    self.logger.debug(f"Sending SSE event: {message.get('id', 'unknown')}")
                    yield {
                        "event": "message",
                        "data": json.dumps(message)
                    }
                    self._message_queue.task_done()
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield {
                        "event": "ping",
                        "data": "{}"
                    }
                    
        return EventSourceResponse(event_generator())

    async def handle_messages(self, request: Request):
        """
        Receives JSON-RPC request.
        Returns 202 Accepted immediately.
        Queues processing.
        """
        try:
            body = await request.json()
        except:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            
        self.logger.info(f"Received JSON-RPC message: {body.get('method')} ID: {body.get('id')}")
        
        # Critical for Gateway: Return 202 immediately
        asyncio.create_task(self._process_rpc_message(body))
        
        return Response(status_code=202)

    async def run_stdio(self):
        """Run the server using stdin/stdout for JSON-RPC."""
        import sys
        
        # Unbuffered stdin/stdout is crucial
        stdin = sys.stdin
        stdout = sys.stdout
        
        self.logger.info("Starting stdio server...")
        
        while True:
            try:
                line = await asyncio.to_thread(stdin.readline)
                if not line:
                    break
                    
                message = json.loads(line)
                
                # We reuse the internal processing logic, but we need to capture the response
                # specifically for this request instead of putting it on the global SSE queue.
                # Refactoring _process_rpc_message to return response instead of queuing would be cleaner,
                # but for minimal invasion, we'll intercept.
                
                # Hack: We'll create a temporary queue for this single request flow or just await the handler directly?
                # _process_rpc_message is async but puts on queue. 
                # Let's refactor _process_rpc_message slightly to be usable by both.
                
                response = await self._process_rpc_message_direct(message)
                if response:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Stdio Error: {e}")
                
    async def _process_rpc_message_direct(self, message: JsonRpcMessage) -> Optional[Dict]:
        """Process RPC and return response dict directly (factored out logic)."""
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
                
                if tool_name in self.tools:
                    handler = self.tools[tool_name]["handler"]
                    result_content = await handler(**tool_args)
                    
                    content_list = []
                    if isinstance(result_content, str):
                        content_list.append({"type": "text", "text": result_content})
                    # Handle other types...
                    elif isinstance(result_content, (dict, list)):
                        content_list.append({"type": "text", "text": json.dumps(result_content, indent=2)})
                    else:
                        content_list.append({"type": "text", "text": str(result_content)})
                        
                    response["result"] = {"content": content_list}
                else:
                    raise Exception(f"Tool not found: {tool_name}")
            
            elif method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.name, "version": self.version},
                    "capabilities": {"tools": {}}
                }
            
            elif method == "notifications/initialized":
                return None
                
            else:
                 # Ensure we don't crash on ping/unknown, checking spec
                 if method.startswith("ping"):
                     return None
                 raise Exception(f"Method not found: {method}")

        except Exception as e:
            self.logger.exception(f"RPC Error: {method}")
            response["error"] = {"code": -32603, "message": str(e)}
            if "result" in response: del response["result"]
            
        return response

    async def _process_rpc_message(self, message: JsonRpcMessage):
        """Legacy wrapper for SSE queue (modified to use direct)."""
        response = await self._process_rpc_message_direct(message)
        if response:
            await self._message_queue.put(response)

    async def handle_health(self):
        return {"status": "healthy"}

    def run(self, port: int = 8000, transport: str = "sse"):
        """
        Run the server.
        Args:
            port: Port for SSE/HTTP mode.
            transport: 'sse' (default) or 'stdio'. Can be overridden by MCP_TRANSPORT env var.
        """
        import os
        env_transport = os.getenv("MCP_TRANSPORT", transport).lower()
        
        if env_transport == "stdio":
            asyncio.run(self.run_stdio())
        else:
            import uvicorn
            # Allow PORT env override for legacy compatibility
            env_port = int(os.getenv("PORT", port))
            uvicorn.run(self.app, host="0.0.0.0", port=env_port, log_level="info")
