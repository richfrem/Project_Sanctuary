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

## Tools (4)

### `code_lint(path, tool="ruff")`
Run linting on a file or directory.

**Args:**
- `path`: Relative path to file or directory
- `tool`: Linting tool to use (ruff, pylint, flake8)

**Returns:**
- Linting results with any issues found

**Example:**
```python
code_lint("mcp_servers/code/server.py", tool="ruff")
```

### `code_format(path, tool="ruff", check_only=False)`
Format code in a file or directory.

**Args:**
- `path`: Relative path to file or directory
- `tool`: Formatting tool to use (ruff, black)
- `check_only`: If True, only check formatting without modifying files

**Returns:**
- Formatting results

**Example:**
```python
code_format("mcp_servers/code/server.py", tool="ruff", check_only=True)
```

### `code_analyze(path)`
Perform static analysis on code.

**Args:**
- `path`: Relative path to file or directory

**Returns:**
- Analysis results with statistics

**Example:**
```python
code_analyze("mcp_servers/code/")
```

### `code_check_tools()`
Check which code quality tools are available.

**Returns:**
- List of available and unavailable tools

**Example:**
```python
code_check_tools()
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
