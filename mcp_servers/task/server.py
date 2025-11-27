"""
Task MCP Server - MCP Protocol Implementation
Exposes task operations via Model Context Protocol
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from pathlib import Path
import json
import sys
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import TaskStatus, TaskPriority


# Initialize server
app = Server("task-mcp")

# Initialize operations
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
task_ops = TaskOperations(PROJECT_ROOT)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="create_task",
            description="Create a new task file in TASKS/ directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title"
                    },
                    "objective": {
                        "type": "string",
                        "description": "What and why of the task"
                    },
                    "deliverables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of concrete outputs"
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of completion conditions"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Critical", "High", "Medium", "Low"],
                        "description": "Task priority (default: Medium)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["backlog", "todo", "in-progress", "done", "blocked"],
                        "description": "Initial status (default: backlog)"
                    },
                    "lead": {
                        "type": "string",
                        "description": "Assigned person/agent (default: Unassigned)"
                    },
                    "dependencies": {
                        "type": "string",
                        "description": "Task dependencies (e.g., 'Requires #012')"
                    },
                    "related_documents": {
                        "type": "string",
                        "description": "Related files/protocols"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional context"
                    },
                    "task_number": {
                        "type": "integer",
                        "description": "Specific task number (auto-generated if not provided)"
                    }
                },
                "required": ["title", "objective", "deliverables", "acceptance_criteria"]
            }
        ),
        Tool(
            name="update_task",
            description="Update an existing task's metadata or content",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_number": {
                        "type": "integer",
                        "description": "Task number to update"
                    },
                    "updates": {
                        "type": "object",
                        "description": "Dictionary of fields to update"
                    }
                },
                "required": ["task_number", "updates"]
            }
        ),
        Tool(
            name="update_task_status",
            description="Change task status (moves file between directories)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_number": {
                        "type": "integer",
                        "description": "Task number"
                    },
                    "new_status": {
                        "type": "string",
                        "enum": ["backlog", "todo", "in-progress", "done", "blocked"],
                        "description": "New status"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about status change"
                    }
                },
                "required": ["task_number", "new_status"]
            }
        ),
        Tool(
            name="get_task",
            description="Retrieve a specific task by number",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_number": {
                        "type": "integer",
                        "description": "Task number to retrieve"
                    }
                },
                "required": ["task_number"]
            }
        ),
        Tool(
            name="list_tasks",
            description="List tasks with optional filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["backlog", "todo", "in-progress", "done", "blocked"],
                        "description": "Filter by status"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Critical", "High", "Medium", "Low"],
                        "description": "Filter by priority"
                    }
                }
            }
        ),
        Tool(
            name="search_tasks",
            description="Search tasks by content (full-text search)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "create_task":
            # Parse arguments
            title = arguments["title"]
            objective = arguments["objective"]
            deliverables = arguments["deliverables"]
            acceptance_criteria = arguments["acceptance_criteria"]
            
            priority = TaskPriority(arguments.get("priority", "Medium"))
            status = TaskStatus(arguments.get("status", "backlog"))
            lead = arguments.get("lead", "Unassigned")
            dependencies = arguments.get("dependencies")
            related_documents = arguments.get("related_documents")
            notes = arguments.get("notes")
            task_number = arguments.get("task_number")
            
            # Create task
            result = task_ops.create_task(
                title=title,
                objective=objective,
                deliverables=deliverables,
                acceptance_criteria=acceptance_criteria,
                priority=priority,
                status=status,
                lead=lead,
                dependencies=dependencies,
                related_documents=related_documents,
                notes=notes,
                task_number=task_number
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result.to_dict(), indent=2)
            )]
        
        elif name == "update_task":
            task_number = arguments["task_number"]
            updates = arguments["updates"]
            
            result = task_ops.update_task(task_number, updates)
            
            return [TextContent(
                type="text",
                text=json.dumps(result.to_dict(), indent=2)
            )]
        
        elif name == "update_task_status":
            task_number = arguments["task_number"]
            new_status = TaskStatus(arguments["new_status"])
            notes = arguments.get("notes")
            
            result = task_ops.update_task_status(task_number, new_status, notes)
            
            return [TextContent(
                type="text",
                text=json.dumps(result.to_dict(), indent=2)
            )]
        
        elif name == "get_task":
            task_number = arguments["task_number"]
            
            task = task_ops.get_task(task_number)
            
            if task is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": f"Task #{task_number:03d} not found"
                    }, indent=2)
                )]
            
            return [TextContent(
                type="text",
                text=json.dumps(task, indent=2)
            )]
        
        elif name == "list_tasks":
            status = TaskStatus(arguments["status"]) if "status" in arguments else None
            priority = TaskPriority(arguments["priority"]) if "priority" in arguments else None
            
            tasks = task_ops.list_tasks(status, priority)
            
            return [TextContent(
                type="text",
                text=json.dumps(tasks, indent=2)
            )]
        
        elif name == "search_tasks":
            query = arguments["query"]
            
            results = task_ops.search_tasks(query)
            
            return [TextContent(
                type="text",
                text=json.dumps(results, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Unknown tool: {name}"
                }, indent=2)
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": str(e)
            }, indent=2)
        )]


async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
