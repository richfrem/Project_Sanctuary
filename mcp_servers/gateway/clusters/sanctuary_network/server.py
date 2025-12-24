#============================================
# mcp_servers/gateway/clusters/sanctuary_network/server.py
# Purpose: Sanctuary Network Cluster - Dual-Transport Entry Point
# Role: Interface Layer (Aggregator Node)
# Status: ADR-066 v1.3 Compliant (SSEServer for Gateway, FastMCP for STDIO)
# Used by: Gateway Fleet (SSE) and Claude Desktop (STDIO)
#============================================

import os
import sys
import logging

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging

# Setup Logging
logger = setup_mcp_logging("project_sanctuary.sanctuary_network")

#============================================
# Tool Schema Definitions (for SSEServer registration)
#============================================
FETCH_URL_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL to fetch content from"}
    },
    "required": ["url"]
}

SITE_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL to check status"}
    },
    "required": ["url"]
}


#============================================
# SSE Transport Implementation (Gateway Mode)
#============================================
def run_sse_server(port: int):
    """Run using SSEServer for Gateway compatibility (ADR-066 v1.3)."""
    from mcp_servers.lib.sse_adaptor import SSEServer
    from mcp_servers.gateway.clusters.sanctuary_network import tools
    
    server = SSEServer("sanctuary_network", version="1.0.0")
    
    server.register_tool("fetch_url", tools.fetch_url, FETCH_URL_SCHEMA)
    server.register_tool("check_site_status", tools.check_site_status, SITE_STATUS_SCHEMA)
    
    logger.info(f"Starting SSEServer on port {port} (Gateway Mode)")
    server.run(port=port, transport="sse")


#============================================
# STDIO Transport Implementation (Local Mode)
#============================================
def run_stdio_server():
    """Run using FastMCP for local development (Claude Desktop)."""
    from fastmcp import FastMCP
    from mcp_servers.gateway.clusters.sanctuary_network.models import FetchUrlRequest, SiteStatusRequest
    from mcp_servers.gateway.clusters.sanctuary_network import tools
    
    mcp = FastMCP(
        "sanctuary_network",
        instructions="""
        Sanctuary Network Cluster.
        - specialized in fetching public web content.
        - specialized in site availability checks.
        """
    )
    
    @mcp.tool()
    async def fetch_url(request: FetchUrlRequest) -> str:
        """Fetch content from a URL via HTTP GET."""
        return await tools.fetch_url(request.url)
    
    @mcp.tool()
    async def check_site_status(request: SiteStatusRequest) -> str:
        """Check if a site is up (HEAD request)."""
        return await tools.check_site_status(request.url)
    
    logger.info("Starting FastMCP server (STDIO Mode)")
    mcp.run(transport="stdio")


#============================================
# Main Execution Entry Point (ADR-066 v1.3 Canonical Selector)
#============================================
def run_server():
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if MCP_TRANSPORT not in {"stdio", "sse"}:
        logger.error(f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}. Must be 'stdio' or 'sse'.")
        sys.exit(1)
    
    if MCP_TRANSPORT == "sse":
        port = int(os.getenv("PORT", 8000))
        run_sse_server(port)
    else:
        run_stdio_server()


if __name__ == "__main__":
    run_server()
