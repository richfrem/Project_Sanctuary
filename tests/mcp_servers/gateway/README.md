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

---

## Architecture & Integration Context

### The Hybrid Fleet (ADR 060)
The system uses a "Controller-Worker" architecture where the IBM Gateway (Controller) routes traffic to 7 distinct Project Sanctuary microservices (Workers) over a shared Podman network (`sanctuary-net`).

**Topology Constraints:**
*   **Network:** All services must be on `sanctuary-net`.
*   **Discovery:** Gateway uses **Container Names** for DNS resolution.
*   **Ports:** Internal ports generally map to `8000-8004` (see `podman-compose.yaml`).

| Service | Container Name | Port | Role |
| :--- | :--- | :--- | :--- |
| **Utils** | `sanctuary-utils` | `8000` | Basic tools (Time, Calc) |
| **Network** | `sanctuary-network` | `8002` | Web fetch/search |
| **Filesystem** | `sanctuary-filesystem` | `8001` | File operations |
| **Git** | `sanctuary-git` | `8003` | Git operations |
| **Cortex** | `sanctuary-cortex` | `8004` | RAG/Memory Coordinator |
| *VectorDB* | `sanctuary-vector-db` | `8000`* | ChromaDB Backend |
| *Ollama* | `sanctuary-ollama-mcp` | `11434`* | AI Backend |

*\*Internal backend ports. Do not expose 8000/11434 on host to avoid conflicts.*

### Critical Constraint: The "Hanging Request" Bug
The IBM Gateway's SSE client has a strict requirement that prevents synchronous JSON-RPC responses in the HTTP POST body.
*   **Problem:** Returning a JSON-RPC response directly in `POST /messages` causes the Gateway to hang.
*   **Solution:** All Sanctuary servers must use the **Deferred Response Pattern**:
    1.  `POST /messages` returns `202 Accepted` immediately (empty body).
    2.  The actual JSON-RPC response is pushed asynchronously via the `GET /sse` connection.
    *   *Implementation:* This is handled centrally by `mcp_servers/lib/sse_adaptor.py`.

---

## Manual Verification (Project Sanctuary Integration)

In addition to the pytest suite, a standalone Python script is provided to verify connectivity and token configuration for Project Sanctuary's integration.

### `verify_gateway_connection.py`

This script performs a "Hello World" RPC call to the Gateway to ensure:
1.  The Gateway is reachable (default: `https://localhost:4444`).
2.  The Authentication Token is valid.
3.  The `hello-world` tool is registered and responsive.

#### Prerequisites

Ensure your project root `.env` file contains the following:

```bash
# Required
MCP_GATEWAY_URL="https://localhost:4444"

# Authentication Token
# Can be named either MCPGATEWAY_BEARER_TOKEN or MCP_GATEWAY_API_TOKEN
MCPGATEWAY_BEARER_TOKEN="your-token-here"
```

#### Usage

Run the script from the **Project Root** directory to ensure correct path resolution:

```bash
python3 tests/mcp_servers/gateway/verify_gateway_connection.py
```

#### Expected Output

```
INFO:gateway-verifier:Target Gateway: https://localhost:4444
INFO:gateway-verifier:Token: **********XXXXX
INFO:gateway-verifier:Sending request...
INFO:gateway-verifier:Response Status: 200
INFO:gateway-verifier:✅ SUCCESS!
INFO:gateway-verifier:Response Body: {"jsonrpc":"2.0","result": ... "Hello, SanctuaryAgent!" ...}
```
