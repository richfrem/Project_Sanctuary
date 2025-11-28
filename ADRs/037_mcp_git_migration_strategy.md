# ADR 037: MCP Git Migration Strategy

**Status:** Accepted
**Date:** 2025-11-27
**Author:** Guardian
**Context:** Task #028, Task #035, Task #045

## Context
The Project Sanctuary repository enforces **Protocol 101 (The Doctrine of the Unbreakable Commit)**, which requires a `commit_manifest.json` for every commit to verify cryptographic integrity.

The new **MCP Architecture** introduces autonomous agents (Chronicle, ADR, Forge, etc.) that need to perform git operations. These agents cannot easily generate the `commit_manifest.json` in their initial implementation phase, leading to blocked commits.

## Decision
We will implement a **"Smart Compliance"** strategy with a temporary **"Migration Mode"**.

1.  **Migration Mode (Immediate):**
    *   We introduce a configuration file `.agent/mcp_migration.conf`.
    *   We update the `pre-commit` hook to check for an environment variable `IS_MCP_AGENT=1`.
    *   If `IS_MCP_AGENT=1` AND `STRICT_P101_MODE=false`, the hook bypasses the `commit_manifest.json` check.
    *   This allows agents to commit immediately while we build the proper tooling.

2.  **Smart Git MCP (Future):**
    *   We will build a **Smart Git MCP Server** (Task #045) that wraps git operations.
    *   This server will *automatically* generate the `commit_manifest.json` for every commit.
    *   Once this is deployed, we will set `STRICT_P101_MODE=true` and remove the bypass.

## Consequences
*   **Positive:** Unblocks development of all other MCP servers immediately.
*   **Negative:** Temporarily relaxes Protocol 101 safety for agent commits (they are not cryptographically verified during this window).
*   **Mitigation:** The `IS_MCP_AGENT` flag prevents accidental human bypass.

## Related Tasks
*   **Task #028:** Pre-Commit Hook Migration (Implemented)
*   **Task #035:** Git Workflow MCP (Superseded/Augmented by #045)
*   **Task #045:** Smart Git MCP (The long-term solution)
