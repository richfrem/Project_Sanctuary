# ADR MCP Server

**Description:** The ADR MCP server provides tools for managing Architecture Decision Records (ADRs) in the `ADRs/` directory. It enforces the canonical ADR schema, validates sequential numbering, and provides search capabilities.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `adr_create` | Create a new ADR with automatic sequential numbering. | `title` (str): ADR title.<br>`context` (str): Problem description.<br>`decision` (str): Decision made.<br>`consequences` (str): Outcomes.<br>`date` (str, optional): Date.<br>`status` (str, optional): Status.<br>`author` (str, optional): Author.<br>`supersedes` (int, optional): Supersedes ADR #. |
| `adr_update_status` | Update the status of an existing ADR. | `number` (int): ADR number.<br>`new_status` (str): New status.<br>`reason` (str): Reason. |
| `adr_get` | Retrieve a specific ADR by number. | `number` (int): ADR number. |
| `adr_list` | List all ADRs with optional status filter. | `status` (str, optional): Filter by status. |
| `adr_search` | Full-text search across all ADRs. | `query` (str): Search query. |

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
"adr": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/adr",
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
pytest tests/mcp_servers/adr/
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `adr_list` appears in the tool list.
3.  **Call Tool:** Execute `adr_list` and verify it returns existing ADRs.

## Architecture

### Overview
This server manages the `ADRs/` directory using a standardized **3-Layer Architecture**:

1.  **Interface Layer (`server.py`):** Handles tool registration and uses FastMCP.
2.  **Business Logic Layer (`operations.py`):** Centralized `ADROperations` class for file I/O and status logic.
3.  **Safety Layer (`validator.py`):** Enforces sequential numbering and valid status transitions.

### Safety Features
- 🔢 **Sequential Integrity:** Auto-assigns numbers to prevent collisions.
- 🔒 **Immutable History:** Prevents deletion; supports replacement via "Superseded" status.
- ✅ **Schema Compliance:** Enforces Context/Decision/Consequences format.
- 🔄 **State Machine:** Validates status transitions (e.g., Proposed -> Accepted).

## Dependencies

- `mcp`
- `fastmcp`
