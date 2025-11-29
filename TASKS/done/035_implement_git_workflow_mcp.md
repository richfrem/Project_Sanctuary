# Task #035: Implement Git Workflow MCP

**Status:** Done  
**Priority:** Medium  
**Estimated Effort:** 2-3 days  
**Dependencies:** Task #028, Shared Infrastructure, **Task #045 (Smart Git MCP)**
**Related ADR:** [ADR 037: MCP Git Migration Strategy](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/ADRs/037_mcp_git_migration_strategy.md)
**Domain:** `project_sanctuary.system.git_workflow`

---

## Objective

Implement Git Workflow MCP Server for safe branch management.
> [!NOTE]
> This task focuses on *branch management* (create, switch, push). The actual *commit logic* is now handled by the **Smart Git MCP (Task #045)** to ensure Protocol 101 compliance. and workflow automation.

---

## Key Features

## Key Features

```typescript
// 1. Start Work
start_feature(task_id: string, description: string) => {
  branch_name: string, // e.g., "feature/task-045-smart-git"
  base_commit: string
}

// 2. Save Progress (Uses Smart Git MCP)
// push_feature() -> handled by Smart Git MCP or simple push

// 3. Finish Work (Cleanup)
finish_feature(branch_name: string) => {
  status: "merged",
  local_branch_deleted: boolean,
  remote_branch_deleted: boolean,
  current_branch: "main"
}

// 4. Sync
sync_main() => {
  pulled_commits: number,
  is_up_to_date: boolean
}
```

---

## Safety Rules

- **Read-only by default** (most operations are status checks)
- **Auto-stash** uncommitted changes before branch switching
- **No destructive operations**: No delete_branch, merge, rebase, force_push
- **User-controlled merges**: PR merges happen on GitHub, not via MCP
- **No history rewriting**: No reset --hard, rebase, amend operations
- **Branch protection**: Cannot switch to or modify protected branches

---

## Excluded Operations (User Must Do Manually)

- Deleting branches (local or remote)
- Merging branches
- Rebasing
- Pulling from remote (to avoid merge conflicts)
- Force pushing
- Resolving merge conflicts

---

**Domain:** `project_sanctuary.system.git_workflow`  
**Class:** `project_sanctuary_system_git_workflow`  
**Risk Level:** MODERATE
