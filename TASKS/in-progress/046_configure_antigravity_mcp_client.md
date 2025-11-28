# Task #046: Configure Antigravity MCP Client

**Status:** In Progress
**Priority:** Medium
**Domain:** `project_sanctuary.system.config`

---

## Objective
Configure the Antigravity agent (and other IDE agents) to natively use the Project Sanctuary MCP servers. This allows the agent to directly call `create_chronicle_entry` or `smart_commit` instead of running raw shell commands.

## Scope
1.  **Client Configuration**: Update `.agent/mcp_config.json` (or equivalent) to register the local MCP servers.
2.  **Tool Exposure**: Ensure the agent has access to the high-level tools from Chronicle, ADR, and Git MCPs.
3.  **Context Awareness**: Configure the agent to read from `mnemonic_cortex` for context.

## Benefits
- **Safety**: Agents use validated tools instead of raw file edits.
- **Consistency**: Enforces protocols (like P101) automatically.
- **Speed**: Reduces the need for multi-step shell commands.

## Implementation Steps
1.  Identify the MCP client configuration location for the current agent runtime.
2.  Create a script `scripts/generate_mcp_config.py` that scans `mcp_servers/` and generates the client config.
3.  Test the integration by having the agent perform a task using only MCP tools.
