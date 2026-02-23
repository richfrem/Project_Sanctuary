# Chronicle MCP Server

**Description:** The Chronicle MCP ensures the integrity of the project's historical record. It enforces strict rules about immutability and classification to maintain a trusted history of events, decisions, and milestones.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `chronicle_create_entry` | Create a new chronicle entry. | `title` (str): Title.<br>`content` (str): Content.<br>`date` (str, optional): Date.<br>`author` (str, optional): Author.<br>`status` (str, optional): Status.<br>`classification` (str, optional): Classification. |
| `chronicle_append_entry` | Alias for create_entry. | Same as create_entry. |
| `chronicle_update_entry` | Update an existing entry (7-day rule applies). | `entry_number` (int): Entry number.<br>`updates` (dict): Updates.<br>`reason` (str): Reason.<br>`override_approval_id` (str, optional): Approval ID. |
| `chronicle_get_entry` | Retrieve a specific entry. | `entry_number` (int): Entry number. |
| `chronicle_list_entries` | List recent entries. | `limit` (int): Limit (default: 10). |
| `chronicle_read_latest_entries` | Alias for list_entries. | Same as list_entries. |
| `chronicle_search` | Search entries by content. | `query` (str): Search query. |

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
"chronicle": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/chronicle",
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
pytest mcp_servers/chronicle/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `chronicle_list_entries` appears in the tool list.
3.  **Call Tool:** Execute `chronicle_list_entries` and verify it returns recent entries.

## Architecture

### Overview
This server manages the `00_CHRONICLE/ENTRIES/` directory.

**Safety Rules:**
1.  **7-Day Modification Window:** Entries older than 7 days are immutable without override.
2.  **Sequential Numbering:** Auto-assigned.
3.  **No Deletion:** Deprecation only.
4.  **Classification:** Mandatory `public`, `internal`, or `confidential`.

## Dependencies

- `mcp`
