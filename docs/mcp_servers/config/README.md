# Config MCP Server Documentation

## Overview

Config MCP manages configuration files in the `.agent/config/` directory. It provides safe read/write operations for MCP and agent configuration.

## Key Concepts

- **Configuration Management:** Centralized config file storage
- **JSON Support:** Automatic JSON validation
- **Safe Operations:** Backup and validation before writes
- **List/Read/Write/Delete:** Complete CRUD operations

## Server Implementation

- **Server Code:** [mcp_servers/config/server.py](../../../mcp_servers/config/server.py)
- **Operations:** [mcp_servers/config/operations.py](../../../mcp_servers/config/operations.py)

## Testing

- **Test Suite:** [tests/mcp_servers/config/](../../../tests/mcp_servers/config/)
- **Status:** ✅ 8/8 tests passing

## Operations

### `config_list`
List all configuration files in .agent/config directory

### `config_read`
Read a configuration file

**Example:**
```python
config_read(filename="mcp_config.json")
```

### `config_write`
Write a configuration file

**Example:**
```python
config_write(
    filename="custom_config.json",
    content='{"setting": "value"}'
)
```

### `config_delete`
Delete a configuration file

## Directory Structure

```
.agent/config/
├── mcp_config.json
├── agent_config.json
└── ...
```

## Status

✅ **Fully Operational** - All operations tested and working
