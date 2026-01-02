# Task MCP Server

**Description:** The Task MCP server manages task files in the `tasks/` directory structure. It provides tools for creating, updating, moving, reading, listing, and searching tasks following the canonical task schema.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `create_task` | Create a new task file. | `title` (str): Title.<br>`objective` (str): Objective.<br>`deliverables` (List[str]): Deliverables.<br>`acceptance_criteria` (List[str]): Criteria.<br>`priority` (str, optional): Priority.<br>`status` (str, optional): Status.<br>`lead` (str, optional): Lead.<br>`dependencies` (str, optional): Deps.<br>`related_documents` (str, optional): Docs.<br>`notes` (str, optional): Notes.<br>`task_number` (int, optional): Number. |
| `update_task` | Update an existing task's metadata or content. | `task_number` (int): Task #.<br>`updates` (dict): Updates. |
| `update_task_status` | Change task status (moves file between directories). | `task_number` (int): Task #.<br>`new_status` (str): New status.<br>`notes` (str, optional): Notes. |
| `get_task` | Retrieve a specific task by number. | `task_number` (int): Task #. |
| `list_tasks` | List tasks with optional filters. | `status` (str, optional): Filter.<br>`priority` (str, optional): Filter. |
| `search_tasks` | Search tasks by content (full-text search). | `query` (str): Search query. |

## Resources

*No resources currently exposed.*

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required
PROJECT_ROOT=/path/to/Project_Sanctuary
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"task": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/task",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}",
    "PROJECT_ROOT": "${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/task/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `list_tasks` appears in the tool list.
3.  **Call Tool:** Execute `list_tasks` and verify it returns tasks.

## Architecture

### Overview
This server manages the `tasks/` directory structure (`backlog`, `todo`, `in-progress`, `done`).

**Safety Rules:**
1.  **Task Number Uniqueness:** Cannot create duplicate task numbers.
2.  **Circular Dependencies:** Validates dependency references exist.
3.  **Schema Compliance:** All tasks must follow `tasks/task_schema.md`.
4.  **File Path Safety:** All operations restricted to `tasks/` directory.
5.  **No Deletion:** tasks cannot be deleted (archive by moving to done).
6.  **No Git Operations:** File operations only.

## Dependencies

- `mcp`
