# TASK: Enable Optional Logging for All MCP Servers

**Status:** Done
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Add optional file-based logging to all MCP servers using the shared logging_utils.py library. Currently only Agent Persona MCP and Council MCP have logging enabled. This task will extend logging to all other MCP servers (Cortex, Code, Protocol, Git, Chronicle, ADR, Task, Config, Forge) to provide consistent debugging capabilities across the entire MCP ecosystem.

## 2. Deliverables

1. Updated MCP server operations files with logging integration
2. Verification that all MCP servers respect MCP_LOGGING flag
3. Documentation of logging configuration in README files

## 3. Acceptance Criteria

- All MCP servers have logging_utils.py integrated
- MCP_LOGGING environment variable controls logging for all servers
- Logs are written to logs/mcp_server.log when enabled
- Documentation updated with logging configuration
