# Chronicle MCP Tests

This directory contains tests for the Chronicle MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Logic, validation, and model behavior.
-   **Dependencies:** None (mocked or pure logic).
-   **Run:** `pytest tests/mcp_servers/chronicle/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** File I/O, directory structure, and operation logic.
-   **Dependencies:** Filesystem (safe via `tmp_path` fixture in `conftest.py`).
-   **Run:** `pytest tests/mcp_servers/chronicle/integration/ -v`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client.
-   **Run:** Use Antigravity or Claude Desktop to call tools:
    -   `chronicle_create_entry`
    -   `chronicle_list_entries`
    -   `chronicle_read_latest_entries`

## Key Files
-   `conftest.py`: Defines `chronicle_root` fixture to ensure tests run in a safe, temporary directory.
