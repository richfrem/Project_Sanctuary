# Chronicle MCP Server

MCP server for managing historical truth entries in `00_CHRONICLE/ENTRIES/`.

## Purpose

The Chronicle MCP ensures the integrity of the project's historical record. It enforces strict rules about immutability and classification to maintain a trusted history of events, decisions, and milestones.

## Tools

### `chronicle_create_entry`
Create a new chronicle entry.
- **Args:** `title`, `content`, `date` (optional), `author` (optional), `status` (optional), `classification` (optional)
- **Returns:** Entry number and file path

### `chronicle_update_entry`
Update an existing entry.
- **Args:** `entry_number`, `updates` (dict), `reason`, `override_approval_id` (optional)
- **Returns:** Updated fields
- **Safety:** Entries older than 7 days require `override_approval_id` to be modified.

### `chronicle_get_entry`
Retrieve a specific entry.
- **Args:** `entry_number`
- **Returns:** Entry details

### `chronicle_list_entries`
List recent entries.
- **Args:** `limit` (optional, default 10)
- **Returns:** List of entries

### `chronicle_search`
Search entries by content.
- **Args:** `query`
- **Returns:** List of matching entries

## Safety Rules

1.  **7-Day Modification Window:** Entries older than 7 days are considered immutable history. Modifying them requires an explicit `override_approval_id`.
2.  **Sequential Numbering:** Entry numbers are auto-assigned and sequential.
3.  **No Deletion:** Entries can be marked as `deprecated` but never deleted.
4.  **Classification:** Entries must be classified as `public`, `internal`, or `confidential`.

## Configuration

Add to `mcp_config.json`:

```json
"chronicle": {
  "displayName": "Chronicle MCP",
  "command": "/path/to/venv/bin/python",
  "args": ["-m", "mcp_servers.document.chronicle.server"],
  "env": {
    "PROJECT_ROOT": "/path/to/project",
    "PYTHONPATH": "/path/to/project"
  }
}
```
