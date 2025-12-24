#============================================
# mcp_servers/orchestrator/server.py
# Purpose: Orchestrator MCP Server.
#          Central mission control for the Agentic swarm.
# Role: Interface Layer
# Used as: Main service entry point.
#============================================

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
from mcp_servers.orchestrator.operations import OrchestratorOperations
from .models import (
    OrchestratorDispatchMissionRequest,
    OrchestratorStrategicCycleRequest,
    CreateCognitiveTaskRequest,
    CreateDevelopmentCycleRequest,
    QueryMnemonicCortexRequest,
    CreateFileWriteTaskRequest,
    CreateGitCommitTaskRequest,
    ListRecentTasksRequest,
    GetTaskResultRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.orchestrator")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.orchestrator",
    instructions="""
    Use this server to orchestrate missions and development cycles.
    - Dispatch high-level missions to specialized agents.
    - Run strategic cycles for synthesis and adaptation.
    - Generate command.json artifacts for Council deliberation.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
ops = OrchestratorOperations(PROJECT_ROOT)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def orchestrator_dispatch_mission(request: OrchestratorDispatchMissionRequest) -> str:
    """Dispatch a mission to an agent."""
    try:
        return ops.dispatch_mission(
            request.mission_id,
            request.objective,
            request.assigned_agent
        )
    except Exception as e:
        logger.error(f"Error in orchestrator_dispatch_mission: {e}")
        raise ToolError(f"Dispatch failed: {str(e)}")

@mcp.tool()
async def orchestrator_run_strategic_cycle(request: OrchestratorStrategicCycleRequest) -> str:
    """Execute a full Strategic Crucible Loop: Ingest -> Synthesize -> Adapt -> Cache."""
    try:
        return ops.run_strategic_cycle(
            request.gap_description,
            request.research_report_path,
            request.days_to_synthesize
        )
    except Exception as e:
        logger.error(f"Error in orchestrator_run_strategic_cycle: {e}")
        raise ToolError(f"Strategic cycle failed: {str(e)}")

@mcp.tool()
async def create_cognitive_task(request: CreateCognitiveTaskRequest) -> dict:
    """Generate a command.json for Council deliberation."""
    try:
        return ops.create_cognitive_task(
            request.description,
            request.output_path,
            request.max_rounds,
            request.force_engine,
            request.max_cortex_queries,
            request.input_artifacts
        )
    except Exception as e:
        logger.error(f"Error in create_cognitive_task: {e}")
        raise ToolError(f"Task creation failed: {str(e)}")

@mcp.tool()
async def create_development_cycle(request: CreateDevelopmentCycleRequest) -> dict:
    """Generate a command.json for a staged development cycle."""
    try:
        return ops.create_development_cycle(
            request.description,
            request.project_name,
            request.output_path,
            request.max_rounds
        )
    except Exception as e:
        logger.error(f"Error in create_development_cycle: {e}")
        raise ToolError(f"Cycle creation failed: {str(e)}")

@mcp.tool()
async def query_mnemonic_cortex(request: QueryMnemonicCortexRequest) -> dict:
    """Generate a command.json for a RAG query task."""
    try:
        return ops.query_mnemonic_cortex(
            request.query,
            request.output_path,
            request.max_results
        )
    except Exception as e:
        logger.error(f"Error in query_mnemonic_cortex: {e}")
        raise ToolError(f"Query task creation failed: {str(e)}")

@mcp.tool()
async def create_file_write_task(request: CreateFileWriteTaskRequest) -> dict:
    """Generate a command.json for writing a file."""
    try:
        return ops.create_file_write_task(
            request.content,
            request.output_path,
            request.description
        )
    except Exception as e:
        logger.error(f"Error in create_file_write_task: {e}")
        raise ToolError(f"File write task creation failed: {str(e)}")

@mcp.tool()
async def create_git_commit_task(request: CreateGitCommitTaskRequest) -> dict:
    """Generate a command.json for a git commit (P101 compliant)."""
    try:
        return ops.create_git_commit_task(
            request.files,
            request.message,
            request.description,
            request.push
        )
    except Exception as e:
        logger.error(f"Error in create_git_commit_task: {e}")
        raise ToolError(f"Commit task creation failed: {str(e)}")

@mcp.tool()
async def get_orchestrator_status() -> dict:
    """Check if the orchestrator is running and healthy."""
    try:
        return ops.get_orchestrator_status()
    except Exception as e:
        logger.error(f"Error in get_orchestrator_status: {e}")
        raise ToolError(f"Status check failed: {str(e)}")

@mcp.tool()
async def list_recent_tasks(request: ListRecentTasksRequest) -> list:
    """List recent tasks from the orchestrator logs/results."""
    try:
        return ops.list_recent_tasks(request.limit)
    except Exception as e:
        logger.error(f"Error in list_recent_tasks: {e}")
        raise ToolError(f"List failed: {str(e)}")

@mcp.tool()
async def get_task_result(request: GetTaskResultRequest) -> dict:
    """Retrieve the result of a specific task."""
    try:
        return ops.get_task_result(request.task_id)
    except Exception as e:
        logger.error(f"Error in get_task_result: {e}")
        raise ToolError(f"Result retrieval failed: {str(e)}")

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
        port = int(port_env) if port_env else 8008
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
