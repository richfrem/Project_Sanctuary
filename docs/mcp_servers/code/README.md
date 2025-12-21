# Code MCP Server Documentation

## Overview

Code MCP provides code analysis, linting, formatting, and file operations for Project Sanctuary. It includes quality checks and automated code improvements.

## Key Concepts

- **Static Analysis:** Code quality metrics and complexity analysis
- **Linting:** Ruff, Pylint, Flake8 support
- **Formatting:** Automatic code formatting with Ruff/Black
- **File Operations:** Safe read/write with automatic backups

## Server Implementation

- **Server Code:** [mcp_servers/code/server.py](../../../mcp_servers/code/server.py)
- **Operations:** [mcp_servers/code/operations.py](../../../mcp_servers/code/operations.py)

## Testing

- **Test Suite:** [tests/mcp_servers/code/](../../../tests/mcp_servers/code/)
- **Status:** ✅ 11/11 tests passing

## Operations

### `code_read`
Read file contents with size limits

### `code_write`
Write/update file with automatic backup

**Example:**
```python
code_write(
    path="mcp_servers/new_server/server.py",
    content="# New MCP Server\n...",
    backup=True,
    create_dirs=True
)
```

### `code_analyze`
Perform static analysis on code

### `code_lint`
Run linting on a file or directory

### `code_format`
Format code with Ruff or Black

### `code_list_files`
List files in a directory with optional pattern

### `code_find_file`
Find files by name or glob pattern

### `code_search_content`
Search for text/patterns in code files

### `code_get_info`
Get file metadata (size, modified date, line count)

### `code_check_tools`
Check which code quality tools are available

## Status

✅ **Fully Operational** - All operations tested and working
