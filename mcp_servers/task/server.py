"""
Task MCP Server - FastMCP Implementation
Exposes task operations via Model Context Protocol using FastMCP
"""

from fastmcp import FastMCP
from pathlib import Path
import sys
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import TaskStatus, TaskPriority

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.task")

# Initialize operations
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
task_ops = TaskOperations(PROJECT_ROOT)


@mcp.tool()
def create_task(
    title: str,
    objective: str,
    deliverables: List[str],
    acceptance_criteria: List[str],
    priority: str = "Medium",
    status: str = "backlog",
    lead: str = "Unassigned",
    dependencies: Optional[str] = None,
    related_documents: Optional[str] = None,
    notes: Optional[str] = None,
    task_number: Optional[int] = None
) -> str:
    """
    Create a new task file in TASKS/ directory.
    
    Args:
        title: Task title
        objective: What and why of the task
        deliverables: List of concrete outputs
        acceptance_criteria: List of completion conditions
        priority: Task priority (default: Medium)
        status: Initial status (default: backlog)
        lead: Assigned person/agent (default: Unassigned)
        dependencies: Task dependencies (e.g., 'Requires #012')
        related_documents: Related files/protocols
        notes: Additional context
        task_number: Specific task number (auto-generated if not provided)
    """
    try:
        result = task_ops.create_task(
            title=title,
            objective=objective,
            deliverables=deliverables,
            acceptance_criteria=acceptance_criteria,
            priority=TaskPriority(priority),
            status=TaskStatus(status),
            lead=lead,
            dependencies=dependencies,
            related_documents=related_documents,
            notes=notes,
            task_number=task_number
        )
        
        return f"Created Task {result.task_number:03d}: {result.file_path}"
    except Exception as e:
        return f"Error creating task: {str(e)}"


@mcp.tool()
def update_task(
    task_number: int,
    updates: Dict[str, Any]
) -> str:
    """
    Update an existing task's metadata or content.
    
    Args:
        task_number: Task number to update
        updates: Dictionary of fields to update
    """
    try:
        result = task_ops.update_task(task_number, updates)
        # Extract updated fields from the updates dict
        updated_fields = list(updates.keys())
        return f"Updated Task {result.task_number:03d}. Fields: {', '.join(updated_fields)}"
    except Exception as e:
        return f"Error updating task: {str(e)}"


@mcp.tool()
def update_task_status(
    task_number: int,
    new_status: str,
    notes: Optional[str] = None
) -> str:
    """
    Change task status (moves file between directories).
    
    Args:
        task_number: Task number
        new_status: New status (backlog, todo, in-progress, complete, blocked)
        notes: Optional notes about status change
    """
    try:
        result = task_ops.update_task_status(
            task_number,
            TaskStatus(new_status),
            notes
        )
        # Extract status info from result
        return f"Updated Task {result.task_number:03d} status to {new_status}"
    except Exception as e:
        return f"Error updating task status: {str(e)}"


@mcp.tool()
def get_task(task_number: int) -> str:
    """
    Retrieve a specific task by number.
    
    Args:
        task_number: Task number to retrieve
    """
    try:
        task = task_ops.get_task(task_number)
        
        if task is None:
            return f"Task #{task_number:03d} not found"
        
        # Build output safely, handling missing fields
        output = []
        output.append(f"Task {task.get('number', task_number):03d}: {task.get('title', 'Untitled')}")
        output.append(f"Status: {task.get('status', 'unknown')}")
        output.append(f"Priority: {task.get('priority', 'unknown')}")
        output.append(f"Lead: {task.get('lead', 'Unassigned')}")
        
        if 'objective' in task and task['objective']:
            output.append(f"\nObjective:\n{task['objective']}")
        
        if 'deliverables' in task and task['deliverables']:
            output.append(f"\nDeliverables:")
            for d in task['deliverables']:
                output.append(f"- {d}")
        
        if 'acceptance_criteria' in task and task['acceptance_criteria']:
            output.append(f"\nAcceptance Criteria:")
            for c in task['acceptance_criteria']:
                output.append(f"- {c}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error retrieving task: {str(e)}"


@mcp.tool()
def list_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None
) -> str:
    """
    List tasks with optional filters.
    
    Args:
        status: Filter by status (backlog, todo, in-progress, done, blocked)
        priority: Filter by priority (Critical, High, Medium, Low)
    """
    try:
        status_filter = TaskStatus(status) if status else None
        priority_filter = TaskPriority(priority) if priority else None
        
        tasks = task_ops.list_tasks(status_filter, priority_filter)
        
        if not tasks:
            filter_desc = []
            if status:
                filter_desc.append(f"status={status}")
            if priority:
                filter_desc.append(f"priority={priority}")
            filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
            return f"No tasks found{filter_str}"
        
        output = [f"Found {len(tasks)} task(s):"]
        for task in tasks:
            output.append(f"- {task['number']:03d}: {task['title']} [{task['status']}] ({task['priority']})")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error listing tasks: {str(e)}"


@mcp.tool()
def search_tasks(query: str) -> str:
    """
    Search tasks by content (full-text search).
    
    Args:
        query: Search query
    """
    try:
        results = task_ops.search_tasks(query)
        
        if not results:
            return f"No tasks found matching '{query}'"
        
        output = [f"Found {len(results)} task(s) matching '{query}':"]
        for task in results:
            output.append(f"- {task['number']:03d}: {task['title']} [{task['status']}]")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error searching tasks: {str(e)}"


if __name__ == "__main__":
    mcp.run()
