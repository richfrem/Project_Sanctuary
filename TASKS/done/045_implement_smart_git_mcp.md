# Task #045: Implement Smart Git MCP Server

**Status:** Done
**Priority:** High
**Domain:** `project_sanctuary.system.git_workflow`

---

## Objective
Create a specialized "Smart Git MCP" that abstracts the complexities of Project Sanctuary's git rules (Protocol 101, `command.json` legacy rules, pre-commit hooks) into a simple, safe interface for other agents.

## Problem
Currently, agents must manually manage `commit_manifest.json` or remember to set `IS_MCP_AGENT=1`. This is error-prone.

## Solution
A Git MCP Server that:
1.  Automatically generates `commit_manifest.json` for every commit (restoring full P101 compliance).
2.  Handles `command.json` validation if needed.
3.  Exposes simple tools like `smart_commit(message, files)`.

## Key Features
```typescript
smart_commit(
  message: string,
  files: string[]
) => {
  commit_hash: string,
  manifest_generated: boolean,
  p101_verified: boolean
}
```

## Implementation Steps
1.  Extend `GitOperations` class in `core/git/`.
2.  Implement manifest generation logic.
3.  Update `mcp_servers/git_workflow/` to use this new logic.
4.  Remove the `IS_MCP_AGENT` bypass from the pre-commit hook (return to Strict Mode).
