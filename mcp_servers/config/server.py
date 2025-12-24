#============================================
# mcp_servers/config/server.py
# Purpose: MCP Server for Configuration Management.
#          Allows reading, writing, and listing agent configuration files.
# Role: Interface Layer
# Used as: Main service entry point for the mcp_servers.config module.
#============================================

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, Union
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.config.operations import ConfigOperations
from .models import (
    ConfigReadRequest,
    ConfigWriteRequest,
    ConfigDeleteRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.config")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.config",
    instructions="""
    Use this server to manage agent configuration files in .agent/config.
    - List configuration files to see what is available.
    - Read configuration to understand agent behavior or settings.
    - Write or update configuration to change system parameters.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
CONFIG_DIR = os.path.join(PROJECT_ROOT, ".agent/config")
ops = ConfigOperations(CONFIG_DIR)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def config_list() -> str:
    """
    List all configuration files in the .agent/config directory.
    """
    try:
        configs = ops.list_configs()
        if not configs:
            return "No configuration files found."
            
        output = [f"Found {len(configs)} configuration files:"]
        for c in configs:
            output.append(f"- {c['name']} ({c['size']} bytes, {c['modified']})")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in config_list: {e}")
        raise ToolError(f"List failed: {str(e)}")

@mcp.tool()
async def config_read(request: ConfigReadRequest) -> str:
    """
    Read a configuration file.
    """
    try:
        content = ops.read_config(request.filename)
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2)
        return str(content)
    except Exception as e:
        logger.error(f"Error in config_read '{request.filename}': {e}")
        raise ToolError(f"Read failed: {str(e)}")

@mcp.tool()
async def config_write(request: ConfigWriteRequest) -> str:
    """
    Write a configuration file.
    """
    try:
        # Try to parse content as JSON if file extension implies it
        if request.filename.endswith('.json'):
            try:
                data = json.loads(request.content)
                path = ops.write_config(request.filename, data)
            except json.JSONDecodeError:
                # Write as raw string if not valid JSON
                path = ops.write_config(request.filename, request.content)
        else:
            path = ops.write_config(request.filename, request.content)
            
        return f"Successfully wrote config to {path}"
    except Exception as e:
        logger.error(f"Error in config_write '{request.filename}': {e}")
        raise ToolError(f"Write failed: {str(e)}")

@mcp.tool()
async def config_delete(request: ConfigDeleteRequest) -> str:
    """
    Delete a configuration file.
    """
    try:
        ops.delete_config(request.filename)
        return f"Successfully deleted config '{request.filename}'"
    except Exception as e:
        logger.error(f"Error in config_delete '{request.filename}': {e}")
        raise ToolError(f"Delete failed: {str(e)}")

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
        port = int(port_env) if port_env else 8006
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
