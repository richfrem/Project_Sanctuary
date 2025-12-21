# MCP Gateway Operations Inventory

**Status:** Implementation Complete / Optimization Phase  
**Source of Truth:** [`GATEWAY_VERIFICATION_MATRIX.md`](./GATEWAY_VERIFICATION_MATRIX.md)

---

## 🚀 Live Deployment Inventory

The following services constitute the active **Fleet Clusters (6 Front-ends / 8 Total Containers)** as defined in ADR 060.

| # | Container | Port (SSE) | Role | Status |
|---|---|---|---|---|
| **1** | `sanctuary_utils` | **8100** | Low-risk (Time, Calc, Search) | ✅ **Live** |
| **2** | `sanctuary_filesystem` | **8101** | File operations (Code Ops) | ✅ **Live** |
| **3** | `sanctuary_network` | **8102** | External HTTP / Site Status | ✅ **Live** |
| **4** | `sanctuary_git` | **8103** | Protocol 101 Git Ops | ✅ **Live** |
| **5** | `sanctuary_cortex` | **8104** | RAG / Forge LLM Cluster | ✅ **Live** |
| **6** | `sanctuary_domain` | **8105** | Logic (Chronicle, ADR, Task) | ✅ **Live** |

### Backend Support Services
These containers support the front-end clusters but are not registered as standalone MCP gateways.

| # | Container | Port | Role | Supporting |
|---|---|---|---|---|
| **7** | `sanctuary_vector_db` | **8110** | ChromaDB Vector Store | `sanctuary_cortex` |
| **8** | `sanctuary_ollama_mcp` | **11434** | LLM Inference Engine | `sanctuary_cortex` |

---

## Operations Reference

### Canonical Operational Tool
The **[Gateway Client Library](../../mcp_servers/gateway/gateway_client.py)** is the primary tool for all gateway operations.

*   **Self-Registration:** `python3 -m mcp_servers.gateway.gateway_client`
*   **Discovery:** Handled automatically during script execution.
*   **Tool Execution:** `from mcp_servers.gateway.gateway_client import execute_mcp_tool`

### Health Checks
*   **Gateway:** `https://localhost:4444/health`
*   **Container Health:** Each container exposes a JSON-RPC `health` tool.

### Management Commands (Podman)
*   **Start Fleet:** `podman-compose up -d`
*   **Stop Fleet:** `podman-compose down`
*   **Restart Service:** `podman restart <container_name>`
*   **Logs:** `podman logs -f sanctuary-gateway`

### Verification
For comprehensive verification status of every individual tool, consult the **Verification Matrix**:
👉 [**GATEWAY_VERIFICATION_MATRIX.md**](./GATEWAY_VERIFICATION_MATRIX.md)
