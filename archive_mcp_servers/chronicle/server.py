#!/usr/bin/env python3
"""
Chronicle Server
=====================================

Purpose:
    MCP Server for the Living Chronicle.
    Manages the creation, retrieval, and updates of chronicle entries.
    Maintains project history and context (00_CHRONICLE/ENTRIES).

Layer: Interface (MCP)

Usage:
    # Run via MCP Config (STDIO)
    python -m mcp_servers.chronicle.server

    # Run via Gateway (SSE)
    PORT=8004 python -m mcp_servers.chronicle.server

Key Functions / MCP Tools:
    - chronicle_create_entry(request): Add new event
    - chronicle_update_entry(request): Modify existing entry
    - chronicle_get_entry(request): Retrieve by ID
    - chronicle_list_entries(request): Show recent history
    - chronicle_search(request): Search content

Related:
    - mcp_servers/chronicle/operations.py
"""

import os
import sys
import json
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.chronicle.operations import ChronicleOperations
from .models import (
    ChronicleCreateRequest,
    ChronicleUpdateRequest,
    ChronicleGetRequest,
    ChronicleListRequest,
    ChronicleSearchRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.chronicle")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.chronicle",
    instructions="""
    Use this server to manage the Living Chronicle.
    - Create new entries for significant events or decisions.
    - Update existing entries with reason and approval if needed.
    - List and search the chronicle to maintain project context.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
CHRONICLE_DIR = os.path.join(PROJECT_ROOT, "00_CHRONICLE/ENTRIES")
ops = ChronicleOperations(CHRONICLE_DIR)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def chronicle_create_entry(request: ChronicleCreateRequest) -> str:
    """
    Create a new chronicle entry.
    """
    try:
        result = ops.create_entry(
            title=request.title,
            content=request.content,
            author=request.author,
            date_str=request.date,
            status=request.status,
            classification=request.classification
        )
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    except Exception as e:
        logger.error(f"Error creating entry: {e}")
        raise ToolError(f"Creation failed: {str(e)}")

@mcp.tool()
async def chronicle_append_entry(request: ChronicleCreateRequest) -> str:
    """
    Append a new entry to the Chronicle (Alias for create_entry).
    """
    try:
        result = ops.create_entry(
            title=request.title,
            content=request.content,
            author=request.author,
            date_str=request.date,
            status=request.status,
            classification=request.classification
        )
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    except Exception as e:
        logger.error(f"Error appending entry: {e}")
        raise ToolError(f"Append failed: {str(e)}")

@mcp.tool()
async def chronicle_update_entry(request: ChronicleUpdateRequest) -> str:
    """
    Update an existing chronicle entry.
    """
    try:
        result = ops.update_entry(
            entry_number=request.entry_number,
            updates=request.updates,
            reason=request.reason,
            override_approval_id=request.override_approval_id
        )
        return f"Updated Chronicle Entry {result['entry_number']}. Fields: {', '.join(result['updated_fields'])}"
    except Exception as e:
        logger.error(f"Error updating entry: {e}")
        raise ToolError(f"Update failed: {str(e)}")

@mcp.tool()
async def chronicle_get_entry(request: ChronicleGetRequest) -> str:
    """
    Retrieve a specific chronicle entry.
    """
    try:
        entry = ops.get_entry(request.entry_number)
        return f"""Entry {entry['number']}: {entry['title']}
Date: {entry['date']}
Author: {entry['author']}
Status: {entry['status']}
Classification: {entry['classification']}

{entry['content']}"""
    except Exception as e:
        logger.error(f"Error retrieving entry: {e}")
        raise ToolError(f"Retrieval failed: {str(e)}")

@mcp.tool()
async def chronicle_list_entries(request: ChronicleListRequest) -> str:
    """
    List recent chronicle entries.
    """
    try:
        entries = ops.list_entries(request.limit)
        if not entries:
            return "No entries found."
            
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error listing entries: {e}")
        raise ToolError(f"List failed: {str(e)}")

@mcp.tool()
async def chronicle_read_latest_entries(request: ChronicleListRequest) -> str:
    """
    Read the latest entries from the Chronicle (Alias for list_entries).
    """
    try:
        entries = ops.list_entries(request.limit)
        if not entries:
            return "No entries found."
            
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error reading latest entries: {e}")
        raise ToolError(f"Read failed: {str(e)}")

@mcp.tool()
async def chronicle_search(request: ChronicleSearchRequest) -> str:
    """
    Search chronicle entries by content.
    """
    try:
        results = ops.search_entries(request.query)
        if not results:
            return f"No entries found matching '{request.query}'"
            
        output = [f"Found {len(results)} entries matching '{request.query}':"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error searching entries: {e}")
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
        port = int(port_env) if port_env else 8004
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
