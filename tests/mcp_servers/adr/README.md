# ADR MCP Tests

This directory contains tests for the ADR MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Validator logic, status transitions, constraints.
-   **Run:** `pytest tests/mcp_servers/adr/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** File I/O for creating/reading/listing ADRs.
-   **Dependencies:** Filesystem (safe via `tmp_path` fixture in `conftest.py`).
-   **Run:** `pytest tests/mcp_servers/adr/integration/ -v`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client.
-   **Run:** Use Antigravity or Claude Desktop to call:
    -   `adr_create`
    -   `adr_list`
    -   `adr_get`
    -   `adr_update_status`

## Key Files
-   `conftest.py`: Defines `adr_root` fixture for safe temp dir testing.
