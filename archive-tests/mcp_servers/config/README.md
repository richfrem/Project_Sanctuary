# Config MCP Tests

This directory contains tests for the Config MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Path validation logic (Security).
-   **Run:** `pytest tests/mcp_servers/config/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** Filesystem operations (Read/Write/List/Delete/Backup).
-   **Dependencies:** Filesystem (safe via `tmp_path` fixture in `conftest.py`).
-   **Run:** `pytest tests/mcp_servers/config/integration/ -v`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client.
-   **Run:** Use Antigravity or Claude Desktop to call:
    -   `config_list`
    -   `config_read`
    -   `config_write`
    -   `config_delete`

## Key Files
-   `conftest.py`: Defines `config_root` fixture for safe temp dir testing.
