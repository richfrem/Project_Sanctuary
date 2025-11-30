# Protocol MCP Server

MCP server for managing system protocols in `01_PROTOCOLS/`.

## Purpose

The Protocol MCP provides structured access to the project's protocol library, ensuring consistent formatting and metadata management.

## Operations

| Operation | Status | Test Suite | Description |
|-----------|--------|------------|-------------|
| `protocol_create` | ✅ | [test_protocol_operations.py](../../tests/test_protocol_operations.py) | Create new protocol with versioning |
| `protocol_update` | ✅ | [test_protocol_operations.py](../../tests/test_protocol_operations.py) | Update protocol (requires version bump for canonical) |
| `protocol_get` | ✅ | [test_protocol_operations.py](../../tests/test_protocol_operations.py) | Retrieve specific protocol by number |
| `protocol_list` | ✅ | [test_protocol_operations.py](../../tests/test_protocol_operations.py) | List protocols with optional filters |
| `protocol_search` | ✅ | [test_protocol_operations.py](../../tests/test_protocol_operations.py) | Full-text search across protocols |
| `protocol_validate_action` | ⚠️ | [test_protocol_operations.py](../../tests/test_protocol_operations.py) | Validate action against protocols |

**Prerequisite Tests:** [test_protocol_validator.py](../../tests/test_protocol_validator.py)

### Tool Details

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
