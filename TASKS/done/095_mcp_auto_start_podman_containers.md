# TASK: MCP Server Auto-Start Podman Containers

**Status:** complete
**Priority:** Low
**Lead:** Unassigned
**Dependencies:** T093 (Containerize Ollama), T094 (Polymorphic Routing)
**Related Documents:** docs/PODMAN_STARTUP_GUIDE.md

---

## 1. Objective

Enhance MCP servers to automatically start required Podman containers if they're not running, improving developer experience and reducing manual setup steps.

## 2. Deliverables

1. **Health Check Functions:** Add container health checks to MCP server startup
2. **Auto-Start Logic:** Implement graceful container startup with retries
3. **Configuration:** Add `.env` flag to enable/disable auto-start behavior
4. **Documentation:** Update MCP SETUP guides with auto-start information

## 3. Acceptance Criteria


## Notes

**Status Change (2025-12-13):** backlog → complete
Implemented shared container management in `mcp_servers/lib/container_manager.py`. Updated `rag_cortex` and `forge_llm` servers to automatically start and verify their required Podman containers (ChromaDB and Ollama) at startup.
