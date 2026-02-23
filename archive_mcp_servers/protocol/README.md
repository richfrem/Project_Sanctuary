# Protocol MCP Server

**Description:** The Protocol MCP provides structured access to the project's protocol library, ensuring consistent formatting, versioning, and metadata management for all system protocols.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `protocol_create` | Create a new protocol. | `number` (int): Protocol number.<br>`title` (str): Title.<br>`status` (str): Status.<br>`classification` (str): Classification.<br>`version` (str): Version.<br>`authority` (str): Authority.<br>`content` (str): Content.<br>`linked_protocols` (List[str], optional): Links. |
| `protocol_update` | Update an existing protocol. | `number` (int): Protocol number.<br>`updates` (dict): Fields to update.<br>`reason` (str): Reason for update. |
| `protocol_get` | Retrieve a specific protocol. | `number` (int): Protocol number. |
| `protocol_list` | List protocols. | `status` (str, optional): Filter by status. |
| `protocol_search` | Search protocols by content. | `query` (str): Search query. |
| `protocol_validate_action` | Validate action against protocols. | `action` (str): Action to validate. |

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
"protocol": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/protocol",
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
pytest mcp_servers/protocol/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `protocol_list` appears in the tool list.
3.  **Call Tool:** Execute `protocol_list` and verify it returns the list of existing protocols.

## Architecture

### Overview
This server manages the lifecycle of protocols stored in `01_PROTOCOLS/`. It enforces:
- **Unique Numbers:** Protocol numbers must be unique.
- **Header Integrity:** All protocols maintain standard header format.
- **No Deletion:** Protocols can be marked as `DEPRECATED` but never deleted.

## Dependencies

- `mcp`
