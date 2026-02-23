# Protocol MCP Tests

This directory contains tests for the Protocol MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Validator logic and constraints.
-   **Dependencies:** None (mocked or pure logic).
-   **Run:** `pytest tests/mcp_servers/protocol/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** File I/O for creating/reading protocols.
-   **Dependencies:** Filesystem (safe via `tmp_path` fixture in `conftest.py`).
-   **Run:** `pytest tests/mcp_servers/protocol/integration/ -v`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client.
-   **Run:** Use Antigravity or Claude Desktop to call:
    -   `protocol_create`
    -   `protocol_list`
    -   `protocol_get`

## Key Files
-   `conftest.py`: Defines `protocol_root` fixture for safe temp dir testing.
