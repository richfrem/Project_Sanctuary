# MCP Gateway Operations Inventory

**Status:** Implementation In Progress (Task 128+)  
**Source of Truth:** [`tests/mcp_servers/gateway/GATEWAY_VERIFICATION_MATRIX.md`](../../../tests/mcp_servers/gateway/GATEWAY_VERIFICATION_MATRIX.md)

---

## 🚀 Live Deployment Inventory

The following services constitute the active "Fleet of 7" Architecture (ADR 060).

| # | Container | Port (SSE) | Role | Status |
|---|---|---|---|---|
| **1** | `sanctuary-utils` | **8100** | Low-risk tools (time, math) | ✅ **Live** |
| **2** | `sanctuary-filesystem` | **8101** | File operations (code_*) | 🔄 **Deploying** |
| **3** | `sanctuary-network` | **8102** | External HTTP access | ⏳ Pending |
| **4** | `sanctuary-git` | **8103** | Git workflow enforcement | ⏳ Pending |
| **5a** | `sanctuary-cortex` | **8104** | RAG / Knowledge Base | ⏳ Pending |
| **5b** | `sanctuary-vector-db` | **8000** | Backend: ChromaDB | ⏳ Pending |
| **5c** | `sanctuary-ollama-mcp` | **11434** | Backend: LLM Inference | ⏳ Pending |

---

## ⚠️ GAP: Domain Logic Cluster
**Container #6 (Planned)** represents critical business logic currently missing from the container fleet.

| Service | Impact | Plan |
|---|---|---|
| **Chronicle MCP** | Logging/Journaling | Merge to C#6 |
| **Protocol MCP** | Rules/Governance | Merge to C#6 |
| **ADR MCP** | Decision Tracking | Merge to C#6 |
| **Task MCP** | Work Management | Merge to C#6 |

---

## Operations Reference

### Health Checks
*   **Gateway:** `https://localhost:4444/health`
*   **Container Health:** Each container exposes a JSON-RPC `health` tool.

### Management Commands
*   **Start Fleet:** `podman-compose up -d`
*   **Stop Fleet:** `podman-compose down`
*   **Logs:** `podman logs -f sanctuary-gateway`

### Verification
For comprehensive verification status of every individual tool, consult the **Verification Matrix**:
👉 [**GATEWAY_VERIFICATION_MATRIX.md**](../../../tests/mcp_servers/gateway/GATEWAY_VERIFICATION_MATRIX.md)
