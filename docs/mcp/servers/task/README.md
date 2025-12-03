# Task MCP Server Documentation

## Overview

Task MCP manages tasks in the `TASKS/` directory, organized by status (backlog, todo, in-progress, done, blocked). It provides operations to create, retrieve, search, and update tasks.

## Key Concepts

- **Status Workflow:** backlog → todo → in-progress → done (or blocked)
- **Priority Levels:** Critical, High, Medium, Low
- **Deliverables:** Concrete outputs expected from the task
- **Acceptance Criteria:** Conditions for task completion

## Server Implementation

- **Server Code:** [mcp_servers/task/server.py](../../../mcp_servers/task/server.py)
- **Operations:** [mcp_servers/task/operations.py](../../../mcp_servers/task/operations.py)
- **Validator:** [mcp_servers/task/validator.py](../../../mcp_servers/task/validator.py)
- **Models:** [mcp_servers/task/models.py](../../../mcp_servers/task/models.py)

## Testing

- **Test Suite:** [tests/mcp_servers/task/](../../../tests/mcp_servers/task/)
- **Status:** ✅ 15/15 tests passing

## Operations

### `create_task`
Create a new task file in TASKS/ directory

**Example:**
```python
create_task(
    title="Implement Protocol 120",
    objective="Create MCP composition pattern protocol",
    deliverables=["Protocol document", "Example implementations"],
    acceptance_criteria=["Protocol reviewed", "Examples tested"],
    priority="High",
    status="todo",
    lead="AI Assistant"
)
```

### `get_task`
Retrieve a specific task by number

### `list_tasks`
List tasks with optional filters (status, priority)

### `search_tasks`
Search tasks by content (full-text search)

### `update_task`
Update an existing task's metadata or content

### `update_task_status`
Change task status (moves file between directories)

## Directory Structure

```
TASKS/
├── backlog/
├── todo/
├── in-progress/
├── done/
└── blocked/
```

## Status

✅ **Fully Operational** - All operations tested and working
