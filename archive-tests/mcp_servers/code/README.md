# Code MCP Tests

This directory contains tests for the Code MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Path validation logic (Security).
-   **Run:** `pytest tests/mcp_servers/code/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** Filesystem operations (Read/Write/List/Find).
-   **Dependencies:** Filesystem (safe via `tmp_path` fixture in `conftest.py`). External tools (ruff) are skipped if missing.
-   **Run:** `pytest tests/mcp_servers/code/integration/ -v`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client.
-   **Run:** Use Antigravity or Claude Desktop to call:
    -   `code_list_files`
    -   `code_read`
    -   `code_write`
    -   `code_analyze`

## Key Files
-   `conftest.py`: Defines `code_root` fixture for safe temp dir testing.
