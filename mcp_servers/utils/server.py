"""
sanctuary-utils MCP Server
Fleet of 7 - Container #1: Low-risk utility tools

Exposes tools via SSE endpoint for Gateway integration.
Implements Guardrail 1 (Fault Containment) and Guardrail 3 (Network Addressing).
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# Import tools (with fault isolation)
# Use relative import for container compatibility
from tools import time_tool
from tools import calculator_tool
from tools import uuid_tool
from tools import string_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanctuary-utils")

# Tool registry - maps tool names to their handlers
TOOL_REGISTRY = {
    # Time tools
    "time.get_current_time": time_tool.get_current_time,
    "time.get_timezone_info": time_tool.get_timezone_info,
    # Calculator tools
    "calculator.calculate": calculator_tool.calculate,
    "calculator.add": calculator_tool.add,
    "calculator.subtract": calculator_tool.subtract,
    "calculator.multiply": calculator_tool.multiply,
    "calculator.divide": calculator_tool.divide,
    # UUID tools
    "uuid.generate_uuid4": uuid_tool.generate_uuid4,
    "uuid.generate_uuid1": uuid_tool.generate_uuid1,
    "uuid.validate_uuid": uuid_tool.validate_uuid,
    # String tools
    "string.to_upper": string_tool.to_upper,
    "string.to_lower": string_tool.to_lower,
    "string.trim": string_tool.trim,
    "string.reverse": string_tool.reverse,
    "string.word_count": string_tool.word_count,
    "string.replace": string_tool.replace,
}

# Tool manifests for registration
TOOL_MANIFESTS = [
    time_tool.TOOL_MANIFEST,
    calculator_tool.TOOL_MANIFEST,
    uuid_tool.TOOL_MANIFEST,
    string_tool.TOOL_MANIFEST,
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    logger.info("🚀 sanctuary-utils starting up...")
    logger.info(f"📦 Registered tools: {list(TOOL_REGISTRY.keys())}")
    # TODO: Implement Guardrail 2 (Self-Registration) in Phase 3
    yield
    logger.info("👋 sanctuary-utils shutting down...")


app = FastAPI(
    title="sanctuary-utils",
    description="Fleet of 7 - Container #1: Low-risk utility tools",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "service": "sanctuary-utils",
        "tools_count": len(TOOL_REGISTRY),
        "tools": list(TOOL_REGISTRY.keys()),
    }


@app.get("/manifest")
async def get_manifest() -> dict[str, Any]:
    """Return tool manifest for Gateway registration."""
    return {
        "server_name": "sanctuary-utils",
        "version": "1.0.0",
        "endpoint": "http://sanctuary-utils:8000/sse",
        "health_check": "http://sanctuary-utils:8000/health",
        "tools": TOOL_MANIFESTS,
    }


async def handle_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Handle a tool call with fault containment (Guardrail 1).
    
    Each tool call is wrapped in try/except to prevent one tool's
    failure from crashing the entire container.
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }
    
    try:
        # Execute the tool with fault isolation
        handler = TOOL_REGISTRY[tool_name]
        result = handler(**arguments)
        return result
    except Exception as e:
        # Guardrail 1: Fault containment - log and return error, don't crash
        logger.error(f"Tool {tool_name} failed: {e}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e),
            "fault_contained": True,
        }


async def sse_event_generator(request: Request):
    """
    SSE event generator for MCP protocol.
    
    Handles incoming tool calls and streams responses back.
    """
    logger.info("SSE connection established")
    
    # Send initial connection message
    yield {
        "event": "connected",
        "data": json.dumps({
            "server": "sanctuary-utils",
            "tools": list(TOOL_REGISTRY.keys()),
        }),
    }
    
    # Keep connection alive and handle tool calls
    # In a full implementation, this would read from the request stream
    # For now, we yield a heartbeat to keep the connection alive
    while True:
        if await request.is_disconnected():
            logger.info("SSE client disconnected")
            break
        
        # Heartbeat to keep connection alive
        yield {
            "event": "heartbeat",
            "data": json.dumps({"status": "alive"}),
        }
        
        # Wait before next heartbeat (adjust as needed)
        import asyncio
        await asyncio.sleep(30)


@app.get("/sse")
async def sse_endpoint(request: Request):
    """
    SSE endpoint for MCP Gateway communication.
    
    This is the main endpoint that the Gateway connects to for tool calls.
    """
    return EventSourceResponse(sse_event_generator(request))


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request) -> JSONResponse:
    """
    Direct tool call endpoint (for testing without SSE).
    
    POST /tools/time.get_current_time
    Body: {"timezone_name": "UTC"}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    result = await handle_tool_call(tool_name, body)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    
    # Run with hot reload for development
    uvicorn.run(
        "mcp_servers.utils.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
