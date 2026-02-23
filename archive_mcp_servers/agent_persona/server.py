#!/usr/bin/env python3
"""
Agent Persona Server
=====================================

Purpose:
    MCP Server for Agent Personas.
    Facilitates role-based agent dispatch and state management.
    Manages custom persona definitions and lifecycle.

Layer: Interface (MCP)

Usage:
    # Run via MCP Config (STDIO)
    python -m mcp_servers.agent_persona.server

    # Run via Gateway (SSE)
    PORT=8002 python -m mcp_servers.agent_persona.server

Key Functions / MCP Tools:
    - persona_dispatch(request): Send task to specific agent
    - persona_list_roles(): List available personas (built-in/custom)
    - persona_get_state(request): Retrieve conversation history
    - persona_reset_state(request): Clear agent memory
    - persona_create_custom(request): Define new role behavior

Related:
    - mcp_servers/agent_persona/operations.py
"""

import os
import sys
import json
from typing import Optional, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pathlib import Path

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.agent_persona.operations import PersonaOperations
from .models import (
    PersonaDispatchParams,
    PersonaRoleParams,
    PersonaCreateCustomParams
)

# 1. Initialize Logging
logger = setup_mcp_logging("agent_persona")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.agent_persona",
    instructions="""
    Use this server to dispatch tasks to specialized agent personas.
    - Use `persona_list_roles` to see available specialized agents.
    - Use `persona_dispatch` to send a task to a specific role.
    - Maintain state across turns if `maintain_state` is True.
    """
)

# 3. Initialize Operations
try:
    root_path = Path(find_project_root())
    persona_ops = PersonaOperations(root_path) 
except Exception as e:
    logger.warning(f"Could not initialize PersonaOperations with project root: {e}")
    persona_ops = PersonaOperations()

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def persona_dispatch(request: PersonaDispatchParams) -> str:
    """
    Dispatch a task or request to a specific AI persona.
    
    Supports state management and customizable model parameters.
    """
    try:
        response = persona_ops.dispatch(
            role=request.role,
            task=request.task,
            context=request.context,
            maintain_state=request.maintain_state,
            engine=request.engine,
            model_name=request.model_name,
            custom_persona_file=request.custom_persona_file
        )
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Error in persona_dispatch: {e}")
        raise ToolError(f"Dispatch failed: {str(e)}")

@mcp.tool()
async def persona_list_roles() -> str:
    """
    List all available agent persona roles (built-in and custom).
    """
    try:
        response = persona_ops.list_roles()
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Error in persona_list_roles: {e}")
        raise ToolError(f"Failed to list roles: {str(e)}")

@mcp.tool()
async def persona_get_state(request: PersonaRoleParams) -> str:
    """
    Retrieve the current conversation history/state for a specific persona.
    """
    try:
        response = persona_ops.get_state(role=request.role)
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Error in persona_get_state: {e}")
        raise ToolError(f"Failed to get state: {str(e)}")

@mcp.tool()
async def persona_reset_state(request: PersonaRoleParams) -> str:
    """
    Clear the conversation history for a specific persona.
    """
    try:
        response = persona_ops.reset_state(role=request.role)
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Error in persona_reset_state: {e}")
        raise ToolError(f"Failed to reset state: {str(e)}")

@mcp.tool()
async def persona_create_custom(request: PersonaCreateCustomParams) -> str:
    """
    Define and save a new custom agent persona.
    """
    try:
        response = persona_ops.create_custom(
            role=request.role,
            persona_definition=request.persona_definition,
            description=request.description
        )
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Error in persona_create_custom: {e}")
        raise ToolError(f"Failed to create persona: {str(e)}")

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
        port = int(port_env) if port_env else 8002
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
