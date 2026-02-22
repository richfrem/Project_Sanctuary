# Task MCP Tests

This directory contains tests for the Task MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** taskstatus, TaskPriority enum logic.
-   **Run:** `pytest tests/mcp_servers/task/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** File I/O for creating/updating/moving tasks.
-   **Dependencies:** Filesystem (safe via `tmp_path` fixture in `conftest.py`).
-   **Run:** `pytest tests/mcp_servers/task/integration/ -v`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client.
-   **Run:** Use Antigravity or Claude Desktop to call:
    -   `create_task`
    -   `update_task`
    -   `update_task_status`
    -   `list_tasks`
    -   `get_task`

## Key Files
-   `conftest.py`: Defines `task_root` fixture for safe temp dir testing.
