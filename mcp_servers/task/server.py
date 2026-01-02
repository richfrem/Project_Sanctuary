#============================================
# mcp_servers/task/server.py
# Purpose: Task MCP Server.
#          Provides tools for managing Project tasks.
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
from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import (
    taskstatus, 
    TaskPriority,
    TaskCreateRequest,
    TaskUpdateRequest,
    TaskUpdateStatusRequest,
    TaskGetRequest,
    TaskListRequest,
    tasksearchRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.task")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.task",
    instructions="""
    Use this server to manage project tasks within the tasks/ directory.
    - Create new tasks with objectives, deliverables, and criteria.
    - Update task metadata and content as work progresses.
    - Change task status to move them through the workflow (backlog -> todo -> in-progress -> done).
    - Query and list tasks to understand project state.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
ops = TaskOperations(PROJECT_ROOT)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def task_create(request: TaskCreateRequest) -> str:
    """Create a new task file in tasks/ directory."""
    try:
        result = ops.create_task(
            title=request.title,
            objective=request.objective,
            deliverables=request.deliverables,
            acceptance_criteria=request.acceptance_criteria,
            priority=TaskPriority(request.priority),
            status=taskstatus(request.status),
            lead=request.lead,
            dependencies=request.dependencies,
            related_documents=request.related_documents,
            notes=request.notes,
            task_number=request.task_number
        )
        if result.status == "error":
            raise ToolError(result.message)
        return f"Created Task {result.task_number:03d}: {result.file_path}"
    except Exception as e:
        logger.error(f"Error in task_create: {e}")
        raise ToolError(f"Creation failed: {str(e)}")

@mcp.tool()
async def task_update(request: TaskUpdateRequest) -> str:
    """Update an existing task's metadata or content."""
    try:
        result = ops.update_task(request.task_number, request.updates)
        if result.status == "error":
            raise ToolError(result.message)
        return f"Updated Task {result.task_number:03d}. Fields: {', '.join(request.updates.keys())}"
    except Exception as e:
        logger.error(f"Error in task_update: {e}")
        raise ToolError(f"Update failed: {str(e)}")

@mcp.tool()
async def task_update_status(request: TaskUpdateStatusRequest) -> str:
    """Change task status (moves file between directories)."""
    try:
        result = ops.update_task_status(
            request.task_number,
            taskstatus(request.new_status),
            request.notes
        )
        if result.status == "error":
            raise ToolError(result.message)
        return f"Updated Task {result.task_number:03d} status to {request.new_status}"
    except Exception as e:
        logger.error(f"Error in task_update_status: {e}")
        raise ToolError(f"Status update failed: {str(e)}")

@mcp.tool()
async def task_get(request: TaskGetRequest) -> str:
    """Retrieve a specific task by number."""
    try:
        task = ops.get_task(request.task_number)
        if task is None:
            raise ToolError(f"Task #{request.task_number:03d} not found")
        
        output = []
        output.append(f"Task {task.get('number', request.task_number):03d}: {task.get('title', 'Untitled')}")
        output.append(f"Status: {task.get('status', 'unknown')}")
        output.append(f"Priority: {task.get('priority', 'unknown')}")
        output.append(f"Lead: {task.get('lead', 'Unassigned')}")
        
        # We could also return the full content or specific fields
        # return task.get('content', '')
        
        # Following existing server.py logic:
        # (Though we could return a dict if we wanted discovery to be easier)
        return task.get('content', "No content available")
        
    except Exception as e:
        logger.error(f"Error in task_get: {e}")
        raise ToolError(f"Retrieval failed: {str(e)}")

@mcp.tool()
async def task_list(request: TaskListRequest) -> str:
    """List tasks with optional filters."""
    try:
        status_filter = taskstatus(request.status) if request.status else None
        priority_filter = TaskPriority(request.priority) if request.priority else None
        
        tasks = ops.list_tasks(status_filter, priority_filter)
        
        if not tasks:
            return "No tasks found matching criteria"
        
        output = [f"Found {len(tasks)} task(s):"]
        for task in tasks:
            output.append(f"- {task['number']:03d}: {task['title']} [{task['status']}] ({task['priority']})")
        
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in task_list: {e}")
        raise ToolError(f"List failed: {str(e)}")

@mcp.tool()
async def task_search(request: tasksearchRequest) -> str:
    """Search tasks by content (full-text search)."""
    try:
        results = ops.search_tasks(request.query)
        
        if not results:
            return f"No tasks found matching '{request.query}'"
        
        output = [f"Found {len(results)} task(s) matching '{request.query}':"]
        for task in results:
            output.append(f"- {task['number']:03d}: {task['title']} [{task['status']}]")
        
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in task_search: {e}")
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
        port = int(port_env) if port_env else 8010
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
