# TASK: Standardize Server Entry Points for Dual-Mode

**Status:** complete
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** 122
**Related Documents:** None

---

## 1. Objective

Refactor all Python MCP server entry points to properly utilize the dual-mode SSEServer. They must detect the execution environment (Stdio vs Port) and launch the appropriate transport method to support both Side-by-Side scenarios.

## 2. Deliverables

1. Updated mcp_servers/rag_cortex/server.py
2. Updated mcp_servers/code/server.py
3. Updated mcp_servers/git/server.py
4. Updated mcp_servers/utils/server.py
5. Updated mcp_servers/network/server.py

## 3. Acceptance Criteria

- All 5 servers (rag_cortex, code, git, utils, network) use the updated SSEServer.run() method
- Servers default to Stdio/Legacy mode if no PORT env var is present
- Servers use PORT env var if present (Gateway mode)

## Notes

**Status Change (2025-12-19):** backlog → in-progress
Starting incremental refactor of servers (Task 123). First target: Utils.

**Status Change (2025-12-19):** in-progress → complete
All servers refactored and verified for dual-mode.
