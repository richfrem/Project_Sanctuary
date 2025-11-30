# Code MCP Server

**Domain:** `project_sanctuary.code`  
**Version:** 1.0.0  
**Status:** Production Ready

---

## Overview

The Code MCP server provides tools for code quality operations including linting, formatting, and static analysis. It integrates with popular Python code quality tools while enforcing safety checks.

**Key Features:**
- ✅ **Safe Path Validation:** Restricts access to project directory
- ✅ **Multiple Tool Support:** Ruff, Black, Pylint, Flake8, Mypy
- ✅ **Check-Only Mode:** Verify formatting without modifying files
- ✅ **Security:** Prevents directory traversal attacks

---

## Tools (10)

### Quality & Analysis

#### `code_lint(path, tool="ruff")`
Run linting on a file or directory.

**Args:**
- `path`: Relative path to file or directory
- `tool`: Linting tool to use (ruff, pylint, flake8)

**Example:**
```python
code_lint("mcp_servers/code/server.py", tool="ruff")
```

#### `code_format(path, tool="ruff", check_only=False)`
Format code in a file or directory.

**Args:**
- `path`: Relative path to file or directory
- `tool`: Formatting tool to use (ruff, black)
- `check_only`: If True, only check formatting without modifying files

**Example:**
```python
code_format("mcp_servers/code/server.py", tool="ruff", check_only=True)
```

#### `code_analyze(path)`
Perform static analysis on code.

**Args:**
- `path`: Relative path to file or directory

**Example:**
```python
code_analyze("mcp_servers/code/")
```

#### `code_check_tools()`
Check which code quality tools are available.

**Example:**
```python
code_check_tools()
```

### File Discovery & Search

#### `code_find_file(name_pattern, directory=".")`
Find files by name or glob pattern.

**Args:**
- `name_pattern`: File name or glob pattern (e.g., "server.py", "*.py")
- `directory`: Directory to search in (default: project root)

**Example:**
```python
code_find_file("server.py")
code_find_file("*.py", "mcp_servers")
```

#### `code_list_files(directory=".", pattern="*", recursive=True)`
List files in a directory with optional pattern.

**Args:**
- `directory`: Directory to list (default: project root)
- `pattern`: Glob pattern for filtering (default: "*")
- `recursive`: If True, search recursively (default: True)

**Example:**
```python
code_list_files("mcp_servers/code", "*.py")
```

#### `code_search_content(query, file_pattern="*.py", case_sensitive=False)`
Search for text/patterns in code files.

**Args:**
- `query`: Text or pattern to search for
- `file_pattern`: File pattern to search in (default: "*.py")
- `case_sensitive`: If True, perform case-sensitive search (default: False)

**Example:**
```python
code_search_content("class CodeOperations")
code_search_content("import FastMCP", "*.py")
```

### File Operations

#### `code_read(path, max_size_mb=10)`
Read file contents.

**Args:**
- `path`: Relative path to file
- `max_size_mb`: Maximum file size in MB (default: 10)

**Example:**
```python
code_read("mcp_servers/code/server.py")
```

#### `code_write(path, content, backup=True, create_dirs=True)`
Write/update file with automatic backup.

**Args:**
- `path`: Relative path to file
- `content`: Content to write
- `backup`: If True, create backup before overwriting (default: True)
- `create_dirs`: If True, create parent directories if needed (default: True)

**Example:**
```python
code_write("new_module.py", "# New module\nprint('Hello')")
```

#### `code_get_info(path)`
Get file metadata.

**Args:**
- `path`: Relative path to file

**Returns:**
- File metadata (size, modified date, line count, language)

**Example:**
```python
code_get_info("mcp_servers/code/server.py")
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ROOT` | Project root directory | `.` |

### Supported Tools

- **Ruff** - Fast Python linter and formatter
- **Black** - Python code formatter
- **Pylint** - Python static code analyzer
- **Flake8** - Python linting tool
- **Mypy** - Static type checker

---

## Testing

```bash
# Run test suite
PYTHONPATH=. python3 tests/mcp_servers/code/test_operations.py
```

**Test Coverage:** 100% (6/6 tests passing)

---

**Maintainer:** Project Sanctuary Team  
**Last Updated:** 2025-11-30
