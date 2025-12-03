# Chronicle MCP Server Documentation

## Overview

Chronicle MCP manages the historical record of Project Sanctuary in the `00_CHRONICLE/` directory. It provides operations to create, retrieve, search, and update chronicle entries.

## Key Concepts

- **Status Management:** Draft → Published → Canonical → Deprecated
- **Classification:** Public, Internal, Confidential
- **Automatic Numbering:** Sequential entry numbering
- **Search:** Full-text search across all entries

## Server Implementation

- **Server Code:** [mcp_servers/chronicle/server.py](../../../mcp_servers/chronicle/server.py)
- **Operations:** [mcp_servers/chronicle/operations.py](../../../mcp_servers/chronicle/operations.py)
- **Validator:** [mcp_servers/chronicle/validator.py](../../../mcp_servers/chronicle/validator.py)
- **Models:** [mcp_servers/chronicle/models.py](../../../mcp_servers/chronicle/models.py)

## Testing

- **Test Suite:** [tests/mcp_servers/chronicle/](../../../tests/mcp_servers/chronicle/)
- **Status:** ✅ 14/14 tests passing

## Operations

### `chronicle_create_entry`
Create a new chronicle entry

**Example:**
```python
chronicle_create_entry(
    title="MCP Test Verification Complete",
    content="Verified all 12 MCP servers...",
    author="AI Assistant",
    status="published",
    classification="internal"
)
```

### `chronicle_get_entry`
Retrieve a specific chronicle entry by number

### `chronicle_list_entries`
List recent chronicle entries

### `chronicle_search`
Full-text search across all chronicle entries

### `chronicle_update_entry`
Update an existing chronicle entry

### Aliases
- `chronicle_append_entry` → `chronicle_create_entry`
- `chronicle_read_latest_entries` → `chronicle_list_entries`

## Directory Structure

```
00_CHRONICLE/
├── 001_project_inception.md
├── 042_mcp_architecture_complete.md
└── ...
```

## Status

✅ **Fully Operational** - All operations tested and working
