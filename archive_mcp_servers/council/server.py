#!/usr/bin/env python3
"""
Council Server
=====================================

Purpose:
    Council MCP Server.
    Exposes Sanctuary Council Orchestrator capabilities.
    Specialized in multi-agent deliberation for cognitive tasks.

Layer: Interface (MCP)

Usage:
    # Run via MCP Config (STDIO)
    python -m mcp_servers.council.server

    # Run via Gateway (SSE)
    PORT=8003 python -m mcp_servers.council.server

Key Functions / MCP Tools:
    - council_dispatch(request): Execute Council deliberation task
    - council_list_agents(): List available agents and status

Related:
    - mcp_servers/council/operations.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.council.operations import CouncilOperations
from mcp_servers.council.models import CouncilDispatchRequest

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.council")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.council",
    instructions="""
    Sanctuary Council Orchestrator.
    - specialized in multi-agent deliberation.
    - Use this for high-level cognitive tasks, auditing, and strategy.
    """
)

# 3. Initialize Operations
council_ops = CouncilOperations()

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def council_dispatch(request: CouncilDispatchRequest) -> Dict[str, Any]:
    """
    Dispatch a task to the Sanctuary Council for multi-agent deliberation.
    
    This is the CORE capability of the Council MCP - multi-agent cognitive processing.
    """
    try:
        result = council_ops.dispatch_task(
            task_description=request.task_description,
            agent=request.agent,
            max_rounds=request.max_rounds,
            force_engine=request.force_engine,
            model_preference=request.model_preference,
            output_path=request.output_path
        )
        return result
    except Exception as e:
        logger.error(f"Error in council_dispatch: {e}")
        raise ToolError(f"Dispatch failed: {str(e)}")

@mcp.tool()
async def council_list_agents() -> List[Dict[str, Any]]:
    """List all available Council agents and their current status."""
    try:
        return council_ops.list_agents()
    except Exception as e:
        logger.error(f"Error in council_list_agents: {e}")
        raise ToolError(f"Listing agents failed: {str(e)}")

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
        port = int(port_env) if port_env else 8003
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
