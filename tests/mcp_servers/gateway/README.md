# Gateway MCP Tests

This directory contains tests for the Gateway MCP server, organized into a 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Internal logic of custom plugins (Allowlist, Auth).
-   **Run:** `pytest tests/mcp_servers/gateway/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** Connectivity with the running Gateway container and external service binding.
-   **Dependencies:** Requires `mcp-gateway` Podman container running on port 4444.
-   **Run:** `pytest tests/mcp_servers/gateway/integration/ -v -m gateway`

### Layer 3: MCP Operations (End-to-End)
-   **Focus:** Full MCP tool execution via Client routed through Gateway.
-   **Run:** Use Antigravity or Claude Desktop to call tools via the Gateway.

## Key Files
-   `conftest.py`: Defines `gateway_url` fixture for integration tests.
