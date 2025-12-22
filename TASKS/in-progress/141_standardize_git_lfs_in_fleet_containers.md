# TASK: Standardize Git LFS in Fleet Containers

**Status:** in-progress
**Priority:** High
**Lead:** Antigravity
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Install git-lfs in essential Fleet containers to support Protocol 101 Git operations.

## 2. Deliverables

1. Updated Dockerfiles with git-lfs installation.
2. Verification command logs.

## 3. Acceptance Criteria

- 'git lfs version' executes successfully inside the sanctuary_git container.
- 'git finish-feature' mcp tool executes without LFS binary errors.

## Notes

**Status Change (2025-12-22):** backlog → in-progress
Starting work on Git LFS standardization inside Fleet containers to resolve 'git-lfs not found' errors in Gateway git tools.
