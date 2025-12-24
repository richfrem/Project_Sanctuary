# TASK: Standardize all MCP servers on FastMCP

**Status:** in-progress
**Priority:** Critical
**Lead:** Antigravity
**Dependencies:** None
**Related Documents:** ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md

---

## 1. Objective

Migrate all Project Sanctuary MCP servers from the custom `SSEServer` to the official `FastMCP` framework to ensure strict protocol compliance with the IBM ContextForge Gateway and resolve tool discovery failures.

## 2. Deliverables

### Core Servers Migration
- [x] Migrate `adr` server (mcp_servers/adr/server.py)
- [x] Migrate `agent_persona` server (mcp_servers/agent_persona/server.py)
- [ ] Migrate `chronicle` server (mcp_servers/chronicle/server.py)
- [ ] Migrate `code` server (mcp_servers/code/server.py)
- [ ] Migrate `config` server (mcp_servers/config/server.py)
- [ ] Migrate `council` server (mcp_servers/council/server.py)
- [ ] Migrate `forge_llm` server (mcp_servers/forge_llm/server.py)
- [ ] Migrate `git` server (mcp_servers/git/server.py)
- [ ] Migrate `orchestrator` server (mcp_servers/orchestrator/server.py)
- [/] Migrate `protocol` server (mcp_servers/protocol/server.py) - *In Progress*
- [ ] Migrate `rag_cortex` server (mcp_servers/rag_cortex/server.py)
- [/] Migrate `task` server (mcp_servers/task/server.py) - *In Progress*
- [/] Migrate `workflow` server (mcp_servers/workflow/server.py) - *In Progress*

### Gateway Clusters Migration
- [ ] Migrate `sanctuary_cortex` cluster
- [ ] Migrate `sanctuary_domain` cluster
- [ ] Migrate `sanctuary_filesystem` cluster
- [ ] Migrate `sanctuary_git` cluster
- [ ] Migrate `sanctuary_network` cluster
- [ ] Migrate `sanctuary_utils` cluster

### Finalization
- [ ] Deprecate and remove `mcp_servers/lib/sse_adaptor.py`
- [ ] Verify all tool discovery via Gateway

## 3. Acceptance Criteria

- All servers pass `python3 -m <server> --help` without runtime errors.
- Tool discovery works for all servers via the Gateway (verified by `gateway_get_capabilities`).
- Dual-transport (SSE/Stdio) is functional.
- No new code uses `SSEServer`.
