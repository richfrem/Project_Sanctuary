#!/usr/bin/env python3
"""
Workflow Server
=====================================

Purpose:
    Workflow MCP Server.
    Provides access to standard operating procedures (SOPs) and workflows.
    Serves as the interface for ensuring process compliance.

Layer: Interface (MCP)

Usage:
    # Run via MCP Config (STDIO)
    python -m mcp_servers.workflow.server

    # Run via Gateway (SSE)
    PORT=8011 python -m mcp_servers.workflow.server

Key Functions / MCP Tools:
    - workflow_get_available_workflows(): List .agent/workflows contents
    - workflow_read_workflow(request): Read specific workflow file

Related:
    - mcp_servers/workflow/operations.py
    - .agent/workflows/
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
from mcp_servers.workflow.operations import WorkflowOperations
from .models import WorkflowReadRequest

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.workflow")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.workflow",
    instructions="""
    Use this server to access and list standard operating procedures (SOPs) and workflows.
    - List all available workflows in the .agent/workflows directory.
    - Read the content of specific workflow files to guide task execution.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
WORKFLOW_DIR = os.path.join(str(PROJECT_ROOT), ".agent/workflows")
ops = WorkflowOperations(Path(WORKFLOW_DIR))

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def workflow_get_available_workflows() -> str:
    """List all available workflows in the .agent/workflows directory."""
    try:
        workflows = ops.list_workflows()
        if not workflows:
            return "No workflows found in .agent/workflows."
        
        output = [f"Found {len(workflows)} available workflow(s):"]
        for wf in workflows:
            turbo = " [TURBO]" if wf.get('turbo_mode') else ""
            output.append(f"- {wf['filename']}{turbo}: {wf['description']}")
        
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in workflow_get_available_workflows: {e}")
        raise ToolError(f"List failed: {str(e)}")

@mcp.tool()
async def workflow_read_workflow(request: WorkflowReadRequest) -> str:
    """Read the content of a specific workflow file."""
    try:
        content = ops.get_workflow_content(request.filename)
        if content is None:
            raise ToolError(f"Workflow '{request.filename}' not found.")
        return content
    except Exception as e:
        logger.error(f"Error in workflow_read_workflow: {e}")
        raise ToolError(f"Read failed: {str(e)}")

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
        port = int(port_env) if port_env else 8011
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
