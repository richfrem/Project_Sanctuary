#============================================
# mcp_servers/adr/server.py
# Purpose: MCP Server for ADR Management.
#          Provides tools for creating, updating, retrieving, and listing ADRs.
# Role: Protocol 122 Enforcement
# Used as: Main service entry point for the mcp_servers.adr module.
#============================================

import os
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.adr.operations import ADROperations
from .models import (
    ADRCreateRequest, 
    ADRUpdateStatusRequest, 
    ADRGetRequest, 
    ADRListRequest, 
    ADRSearchRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("adr")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.document.adr",
    instructions="""
    Use this server to manage Architecture Decision Records (ADRs).
    - Use `adr_create` to document new architectural decisions.
    - Use `adr_list` to see existing decisions.
    - Use `adr_get` to read full details of a specific decision.
    """
)

# 3. Initialize ADR operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
ADRS_DIR = os.path.join(PROJECT_ROOT, "ADRs")
adr_ops = ADROperations(ADRS_DIR)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
def adr_create(request: ADRCreateRequest) -> str:
    """
    Create a new Architecture Decision Record (ADR).
    
    This follows the Protocol 122 standard for documenting significant 
    technical decisions in the Project Sanctuary ecosystem.
    """
    try:
        result = adr_ops.create_adr(
            title=request.title,
            context=request.context,
            decision=request.decision,
            consequences=request.consequences,
            date=request.date,
            status=request.status,
            author=request.author,
            supersedes=request.supersedes
        )
        return f"Successfully created ADR {result['adr_number']:03d}: {result['file_path']}"
    except Exception as e:
        logger.error(f"Failed to create ADR: {str(e)}")
        raise ToolError(f"Operation failed: {str(e)}")

@mcp.tool()
def adr_update_status(request: ADRUpdateStatusRequest) -> str:
    """
    Update the status of an existing ADR.
    
    Use this when a decision moves from 'proposed' to 'accepted', 
    or when a decision is 'deprecated' or 'superseded' by a newer ADR.
    """
    try:
        result = adr_ops.update_adr_status(request.number, request.new_status, request.reason)
        return (
            f"Updated ADR {result['adr_number']:03d}: "
            f"{result['old_status']} → {result['new_status']} "
            f"(Reason: {request.reason})"
        )
    except Exception as e:
        logger.error(f"Failed to update ADR status: {str(e)}")
        raise ToolError(f"Operation failed: {str(e)}")

@mcp.tool()
def adr_get(request: ADRGetRequest) -> str:
    """
    Retrieve full details of a specific ADR by its number.
    
    Provides the context, decision, and consequences of the record.
    """
    try:
        adr = adr_ops.get_adr(request.number)
        return (
            f"ADR {adr['number']:03d}: {adr['title']}\n"
            f"Status: {adr['status']}\n"
            f"Date: {adr['date']}\n"
            f"Author: {adr['author']}\n\n"
            f"Context:\n{adr['context']}\n\n"
            f"Decision:\n{adr['decision']}\n\n"
            f"Consequences:\n{adr['consequences']}"
        )
    except Exception as e:
        logger.error(f"Failed to get ADR {request.number}: {str(e)}")
        raise ToolError(f"Operation failed: {str(e)}")

@mcp.tool()
def adr_list(request: ADRListRequest) -> str:
    """
    List all ADRs in the repository.
    
    Optionally filter by status to narrow down records.
    """
    try:
        adrs = adr_ops.list_adrs(request.status)
        if not adrs:
            return "No ADRs found" + (f" with status '{request.status}'" if request.status else "")
        
        result = f"Found {len(adrs)} ADR(s)" + (f" with status '{request.status}'" if request.status else "") + ":\n\n"
        for adr in adrs:
            result += f"ADR {adr['number']:03d}: {adr['title']} [{adr['status']}] ({adr['date']})\n"
        
        return result
    except Exception as e:
        logger.error(f"Failed to list ADRs: {str(e)}")
        raise ToolError(f"Operation failed: {str(e)}")

@mcp.tool()
def adr_search(request: ADRSearchRequest) -> str:
    """
    Perform a full-text search across all ADRs.
    
    Returns matching ADR numbers, titles, and relevant snippets.
    """
    try:
        results = adr_ops.search_adrs(request.query)
        if not results:
            return f"No ADRs found matching '{request.query}'"
        
        output = f"Found {len(results)} ADR(s) matching '{request.query}':\n\n"
        for result in results:
            output += f"ADR {result['number']:03d}: {result['title']}\n"
            for match in result['matches']:
                output += f"  - {match}\n"
            output += "\n"
        
        return output
    except Exception as e:
        logger.error(f"Failed to search ADRs: {str(e)}")
        raise ToolError(f"Operation failed: {str(e)}")

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
        port = int(port_env) if port_env else 8001
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
