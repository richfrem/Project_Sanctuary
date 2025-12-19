#!/usr/bin/env python3
"""
Sanctuary Gateway Client Bridge
==============================

Acts as a local MCP Server (over Stdio) that proxies requests to the
remote Sanctuary Gateway (over SSE).

Usage:
  python mcp_gateway_client.py

Configuration (Env Vars):
  MCP_GATEWAY_URL: URL of the Gateway SSE endpoint (default: http://localhost:4444/sse)
  MCP_GATEWAY_API_TOKEN: API Token for authentication
"""

import asyncio
import os
import sys
import logging
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr to avoid corrupting stdio transport
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("gateway-bridge")

# Configuration
GATEWAY_URL = os.environ.get("MCP_GATEWAY_URL", "http://localhost:4444/sse")
API_TOKEN = os.environ.get("MCP_GATEWAY_API_TOKEN")

if not API_TOKEN:
    logger.error("MCP_GATEWAY_API_TOKEN environment variable is required")
    sys.exit(1)

# Initialize FastMCP Server (The "Face" to Antigravity)
mcp = FastMCP("Sanctuary Gateway Bridge")

@mcp.tool()
async def gateway_proxy_tool(tool_name: str, arguments: dict) -> str:
    """
    Proxy a tool call to the Sanctuary Gateway.
    
    This is a meta-tool. In a real dynamic proxy, we would dynamically
    register tools based on the Gateway's list. For this bridge,
    we might need a different approach if FastMCP requires static tools.
    
    However, standard MCP clients (like Antigravity) query `list_tools`.
    We need to intercept `list_tools` and `call_tool`.
    FastMCP abstracts this, which might be too high-level if we want
    dynamic proxying.
    
    Let's switch to low-level Server implementation for full transparency.
    """
    return "Not implemented in this mode"

# --- Low Level Implementation ---

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

async def run_bridge():
    logger.info(f"Connecting to Gateway at {GATEWAY_URL}...")
    
    # 1. Connect to Gateway (as a Client)
    async with sse_client(GATEWAY_URL, headers={"Authorization": f"Bearer {API_TOKEN}"}) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("Connected to Gateway!")
            
            # 2. Fetch Tools from Gateway to expose them locally
            # Note: In a robust bridge, we'd refresh this periodically or on-demand.
            gateway_tools_result = await session.list_tools()
            gateway_tools = gateway_tools_result.tools
            logger.info(f"Discovered {len(gateway_tools)} tools from Gateway")
            
            # 3. Initialize Local Server (to serve detailed tool list to Antigravity)
            # We explicitly define the capabilities we proxy.
            
            app = Server("Sanctuary Gateway Bridge")
            
            @app.list_tools()
            async def list_tools() -> list[types.Tool]:
                # Dynamic fetch (or cached)
                # We fetch fresh every time to be truly dynamic
                logger.info("Antigravity requested tool list - fetching from Gateway...")
                fresh_result = await session.list_tools()
                return fresh_result.tools

            @app.call_tool()
            async def call_tool(
                name: str, arguments: Any
            ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
                logger.info(f"Proxying tool call: {name}")
                try:
                    # Forward the call to the Gateway
                    result = await session.call_tool(name, arguments)
                    return result.content
                except Exception as e:
                    logger.error(f"Gateway tool call failed: {e}")
                    return [types.TextContent(type="text", text=f"Gateway Error: {str(e)}")]

            # 4. Run Stdio Server
            logger.info("Starting Stdio Server for Antigravity...")
            async with stdio_server() as (read_stream, write_stream):
                await app.run(
                    read_stream,
                    write_stream,
                    app.create_initialization_options(),
                )

if __name__ == "__main__":
    try:
        asyncio.run(run_bridge())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Bridge crashed")
