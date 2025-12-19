# Side-by-Side Architecture: Port Management Strategy

## Objective
Support two distinct deployment modes without conflict:
1.  **Legacy/Direct Mode:** Standard MCP servers running locally (often via start scripts).
2.  **Gateway Mode:** Containerized microservices building the "Hybrid Fleet" routed via IBM ContextForge Gateway.

## Port Configuration (Truth Source)

The following port assignments are defined for the side-by-side operation.

### Legacy / Direct Mode
*Standard local ports used when running servers individually or via `start_sanctuary.sh`*

*   **RAG Cortex:** `8000` (`MCP_SERVER_RAG_CORTEX`)
*   **Code:** `8001` (`MCP_SERVER_CODE`)
*   **Network:** `8002` (`MCP_SERVER_NETWORK`)
*   **Git:** `8003` (`MCP_SERVER_GIT`)
*   **RAG Cortex (Alt/Future):** `8004` (`MCP_SERVER_RAG_CORTEX`) - *Note: Disambiguation needed if 8000 is occupied.*
*   **Vector DB (Chroma):** `8000` (`MCP_SERVER_VECTOR_DB` - *Conflict with Cortex legacy?*)
*   **Ollama:** `11434` (`MCP_SERVER_OLLAMA`)

### Gateway Mode (Podman/Docker)
*Containerized fleet accessible via Host Ports or Gateway*

*   **Gateway:** `4444` (`MCP_SERVER_MCPGATEWAY`) - *The single entry point for clients.*
*   **Sanctuary Utils:** `8100` (`MCP_SERVER_SANCTUARY_UTILS`)
*   **Sanctuary Filesystem:** `8101` (Mapped to internal `8001`)
*   **Sanctuary Network:** `8102` (Mapped to internal `8002`)
*   **Sanctuary Git:** `8103` (Mapped to internal `8003`)
*   **Sanctuary Cortex:** `8104` (Mapped to internal `8004`)
*   **Hello World:** `8005` (`MCP_SERVER_HELLO_WORLD`)

## Configuration Strategy

**"What I will do only is choose to make changes to the claude desktop mcp config or gemini mcp server config to choose to use old approach or new with gateway"**

### Toggle Mechanism
1.  **For Gateway Usage:**
    *   Client Config points to: `http://localhost:4444/sse` (and uses headers/arguments for routing).
    *   *OR* Client Config points to specific Host Mapped ports (e.g., `8100` for Utils) for direct container access.

2.  **For Legacy Usage:**
    *   Client Config points to: `http://localhost:8000`, `8001`, etc.
    *   *OR* Client Config uses `stdio` command execution (e.g., `python -m mcp_servers.start`).

## Verification Requirements
*   Ensure starting the Gateway Fleet (Podman) does not seize ports `8000-8003` on the host, preventing Legacy usage.
    *   *Check:* Utils maps `8100:8000`. Safe? Yes, if Utils internal is 8000, Host is 8100.
    *   *Check:* Cortex maps `8104:8004`. Safe.
