# Task #035: Implement Git Workflow MCP

**Status:** Backlog  
**Priority:** Medium  
**Estimated Effort:** 2-3 days  
**Dependencies:** Task #028, Shared Infrastructure  
**Domain:** `project_sanctuary.system.git_workflow`

---

## Objective

Implement Git Workflow MCP server for safe branch management and workflow automation.

---

## Key Features

```typescript
create_feature_branch(branch_name, base_branch?)
switch_branch(branch_name, stash_changes?)
push_current_branch(set_upstream?)
get_repo_status()
list_branches()
compare_branches(source, target)
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
