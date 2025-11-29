# Protocol MCP Server

MCP server for managing system protocols in `01_PROTOCOLS/`.

## Purpose

The Protocol MCP provides structured access to the project's protocol library, ensuring consistent formatting and metadata management.

## Tools

### `protocol_create`
Create a new protocol.
- **Args:** `number`, `title`, `status`, `classification`, `version`, `authority`, `content`, `linked_protocols` (optional)
- **Returns:** Protocol number and file path

### `protocol_update`
Update an existing protocol.
- **Args:** `number`, `updates` (dict), `reason`
- **Returns:** Updated fields

### `protocol_get`
Retrieve a specific protocol.
- **Args:** `number`
- **Returns:** Protocol details

### `protocol_list`
List protocols.
- **Args:** `status` (optional filter)
- **Returns:** List of protocols

### `protocol_search`
Search protocols by content.
- **Args:** `query`
- **Returns:** List of matching protocols

## Safety Rules

1.  **Unique Numbers:** Protocol numbers must be unique.
2.  **Header Integrity:** All protocols maintain standard header format.
3.  **No Deletion:** Protocols can be marked as `DEPRECATED` but never deleted.

## Configuration

Add to `mcp_config.json`:

```json
"protocol": {
  "displayName": "Protocol MCP",
  "command": "/path/to/venv/bin/python",
  "args": ["-m", "mcp_servers.system.protocol.server"],
  "env": {
    "PROJECT_ROOT": "/path/to/project",
    "PYTHONPATH": "/path/to/project"
  }
}
```
