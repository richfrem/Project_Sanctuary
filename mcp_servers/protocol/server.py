#!/usr/bin/env python3
"""
Protocol Server
=====================================

Purpose:
    Protocol MCP Server.
    Provides tools for managing the Protocol Library (01_PROTOCOLS).
    Handles lifecycle of Standard Operating Procedures (SOPs).

Layer: Interface (MCP)

Usage:
    # Run via MCP Config (STDIO)
    python -m mcp_servers.protocol.server

    # Run via Gateway (SSE)
    PORT=8009 python -m mcp_servers.protocol.server

Key Functions / MCP Tools:
    - protocol_create(request): Create new protocol
    - protocol_update(request): Update existing protocol
    - protocol_get(request): Retrieve protocol content
    - protocol_list(request): List protocols by status
    - protocol_search(request): Search protocol content

Related:
    - mcp_servers/protocol/operations.py
    - 01_PROTOCOLS/ directory
"""

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.protocol.operations import ProtocolOperations
from .models import (
    ProtocolCreateRequest,
    ProtocolUpdateRequest,
    ProtocolGetRequest,
    ProtocolListRequest,
    ProtocolSearchRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.protocol")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.protocol",
    instructions="""
    Use this server to manage the project's Standard Operating Procedures (SOPs) and protocols.
    - Create new protocols for repeatable processes or standards.
    - Update existing protocols with clear justification.
    - Search protocols to maintain methodological consistency.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
PROTOCOL_DIR = os.path.join(str(PROJECT_ROOT), "01_PROTOCOLS")
ops = ProtocolOperations(PROTOCOL_DIR)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def protocol_create(request: ProtocolCreateRequest) -> str:
    """Create a new protocol."""
    try:
        result = ops.create_protocol(
            request.number,
            request.title,
            request.status,
            request.classification,
            request.version,
            request.authority,
            request.content,
            request.linked_protocols
        )
        return f"Created Protocol {result['protocol_number']}: {result['file_path']}"
    except Exception as e:
        logger.error(f"Error in protocol_create: {e}")
        raise ToolError(f"Creation failed: {str(e)}")

@mcp.tool()
async def protocol_update(request: ProtocolUpdateRequest) -> str:
    """Update an existing protocol."""
    try:
        result = ops.update_protocol(
            request.number,
            request.updates,
            request.reason
        )
        return f"Updated Protocol {result['protocol_number']}. Fields: {', '.join(result['updated_fields'])}"
    except Exception as e:
        logger.error(f"Error in protocol_update: {e}")
        raise ToolError(f"Update failed: {str(e)}")

@mcp.tool()
async def protocol_get(request: ProtocolGetRequest) -> str:
    """Retrieve a specific protocol."""
    try:
        protocol = ops.get_protocol(request.number)
        return f"""Protocol {protocol['number']}: {protocol['title']}
Status: {protocol['status']}
Classification: {protocol['classification']}
Version: {protocol['version']}
Authority: {protocol['authority']}
Linked Protocols: {protocol.get('linked_protocols', 'None')}

{protocol['content']}"""
    except Exception as e:
        logger.error(f"Error in protocol_get: {e}")
        raise ToolError(f"Retrieval failed: {str(e)}")

@mcp.tool()
async def protocol_list(request: ProtocolListRequest) -> str:
    """List protocols with optional status filter."""
    try:
        protocols = ops.list_protocols(request.status)
        if not protocols:
            return "No protocols found."
            
        output = [f"Found {len(protocols)} protocol(s):"]
        for p in protocols:
            output.append(f"- {p['number']:03d}: {p['title']} [{p['status']}] v{p['version']}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in protocol_list: {e}")
        raise ToolError(f"List failed: {str(e)}")

@mcp.tool()
async def protocol_search(request: ProtocolSearchRequest) -> str:
    """Search protocols by content."""
    try:
        results = ops.search_protocols(request.query)
        if not results:
            return f"No protocols found matching '{request.query}'"
            
        output = [f"Found {len(results)} protocol(s) matching '{request.query}':"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in protocol_search: {e}")
        raise ToolError(f"Search failed: {str(e)}")

#============================================
# Main Execution Entry Point
#============================================

if __name__ == "__main__":
    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Local/CLI Mode)
    port_env = get_env_variable("PORT", required=False)
    transport = "sse" if port_env else "stdio"
    
    if transport == "sse":
        port = int(port_env) if port_env else 8009
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
