# Config MCP Server

**Domain:** `project_sanctuary.config`  
**Version:** 1.0.0  
**Status:** Production Ready

---

## Overview

The Config MCP server provides tools for managing configuration files within the `.agent/config/` directory. It ensures safe access to configuration data with path validation, automatic backups, and support for JSON/YAML formats.

**Key Features:**
- ✅ **Safe Path Validation:** Restricts access to `.agent/config/` directory
- ✅ **Automatic Backups:** Creates timestamped backups before overwriting files
- ✅ **Format Support:** Native JSON and YAML handling
- ✅ **Security:** Prevents directory traversal attacks

---

## Tools (4)

### `config_list()`
List all configuration files in the `.agent/config` directory.

**Returns:**
- List of files with sizes and modification times

**Example:**
```python
config_list()
# Returns:
# Found 2 configuration files:
# - mcp_config.json (1024 bytes, Mon Nov 30 10:00:00 2025)
# - agent_settings.yaml (512 bytes, Mon Nov 30 10:05:00 2025)
```

### `config_read(filename)`
Read a configuration file.

**Args:**
- `filename`: Name of the config file (e.g., `mcp_config.json`)

**Returns:**
- Content of the file (JSON formatted string if valid JSON/YAML)

**Example:**
```python
config_read("mcp_config.json")
```

### `config_write(filename, content)`
Write a configuration file with automatic backup.

**Args:**
- `filename`: Name of the config file
- `content`: Content to write (string or JSON string)

**Returns:**
- Status message with path to written file

**Example:**
```python
config_write("app_settings.json", '{"theme": "dark", "debug": true}')
```

### `config_delete(filename)`
Delete a configuration file.

**Args:**
- `filename`: Name of the config file to delete

**Returns:**
- Status message

**Example:**
```python
config_delete("temp_config.json")
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ROOT` | Project root directory | `.` |

### Directory Structure

```
PROJECT_ROOT/
├── .agent/
│   └── config/          # Managed configuration directory
│       ├── mcp_config.json
│       ├── ...
│       └── *.bak        # Automatic backups
```

---

## Testing

```bash
# Run test suite
PYTHONPATH=. python3 tests/mcp_servers/config/test_operations.py
```

---

**Maintainer:** Project Sanctuary Team  
**Last Updated:** 2025-11-30
