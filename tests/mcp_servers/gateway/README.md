# Gateway MCP Tests

This directory contains tests for the **External Gateway MCP** (decoupled Podman service), organized into the standard 3-layer pyramid.

## Structure

### Layer 1: Unit Tests (`unit/`)
-   **Focus:** Internal logic of custom plugins (if any are vendored for testing).
-   **Run:** `pytest tests/mcp_servers/gateway/unit/ -v`

### Layer 2: Integration Tests (`integration/`)
-   **Focus:** Connectivity with the running Gateway container (Black Box testing).
-   **Dependencies:** Requires `sanctuary-gateway` Podman container running on port 4444.
-   **Run:** `pytest tests/mcp_servers/gateway/integration/ -v -m gateway`

### Layer 3: E2E Tests (`e2e/`)
-   **Focus:** Full MCP tool execution via Client routed through Gateway.
-   **Run:** Use Antigravity or Claude Desktop to call tools via the Gateway.

## Key Files
-   `integration/test_gateway_blackbox.py`: Pulse, Circuit Breaker, and Handshake checks for the external service.

## External Service
The Gateway runs as a separate service at `https://localhost:4444` and is managed via Podman. See ADR 058 for the decoupling rationale.
