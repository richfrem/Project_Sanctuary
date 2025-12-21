# Cluster: sanctuary_git

**Role:** Git workflow orchestration and Protocol 101 enforcement.  
**Port:** 8103  
**Front-end Cluster:** ✅ Yes

## Overview
The `sanctuary_git` cluster handles all version control operations. It resolves the `mcp-server-git` vs standard `git` naming collision by nesting all operations under the `sanctuary_git-` gateway prefix.

## Verification Specs (Tier 3: Bridge)
*   **Target:** Gateway Bridge & RPC Routing
*   **Method:** `pytest tests/mcp_servers/gateway/clusters/sanctuary_git/test_gateway.py`
*   **Integrity**: Enforces the **Doctrine of Absolute Stability** (Protocol 101).

## Tool Inventory & Legacy Mapping
| Function | Gateway Tool Name | Legacy Operation | T1/T2 Method |
| :--- | :--- | :--- | :--- |
| **Status** | `sanctuary_git-git-get-status` | `git_get_status` | `pytest tests/mcp_servers/git/` |
| **Diff** | `sanctuary_git-git-diff` | `git_diff` | |
| **Log** | `sanctuary_git-git-log` | `git_log` | |
| **Branch** | `sanctuary_git-git-start-feature` | `git_start_feature` | |
| **Add** | `sanctuary_git-git-add` | `git_add` | |
| **Commit** | `sanctuary_git-git-smart-commit` | `git_smart_commit` | |
| **Push** | `sanctuary_git-git-push-feature` | `git_push_feature` | |
| **Finish** | `sanctuary_git-git-finish-feature` | `git_finish_feature` | |

## Documentation Reference
- **ADR 037**: MCP Git Migration Strategy.
- **Protocol 101**: Doctrine of the Unbreakable Commit.
