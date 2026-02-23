# Git Workflow MCP - Safety Features Documentation

**Version:** 1.0  
**Last Updated:** 2025-11-30  
**Status:** Production Ready

---

## Overview

The Git Workflow MCP implements a **strict safety system** to prevent dangerous git operations and enforce a disciplined feature branch workflow. This document details all safety features, their rationale, and test coverage.

---

## Safety Philosophy

### Core Principles

1. **Never Commit to Main:** All development must occur on feature branches
2. **One Feature at a Time:** Only one active feature branch allowed
3. **Verify Before Trust:** All operations verify state before proceeding
4. **Merge Before Delete:** Feature branches can only be deleted after PR merge
5. **Clean State Required:** Critical operations require clean working directory

### Removed Operations

- **`git_sync_main`** - Removed entirely (unsafe standalone operation)
  - **Rationale:** Agents were pulling main prematurely, before PR merge
  - **Alternative:** Sync happens automatically in `git_finish_feature` after merge verification

---

## Operation Safety Matrix

| Operation | Main Block | Feature Check | Clean State | Merge Verify | Idempotent |
|-----------|------------|---------------|-------------|--------------|------------|
| `git_get_status` | N/A | N/A | N/A | N/A | ✅ |
| `git_diff` | N/A | N/A | N/A | N/A | ✅ |
| `git_log` | N/A | N/A | N/A | N/A | ✅ |
| `git_start_feature` | N/A | ✅ | ✅ | N/A | ✅ |
| `git_add` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_smart_commit` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_push_feature` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_finish_feature` | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## Detailed Safety Checks

### 1. `git_start_feature`

**Purpose:** Create or switch to a feature branch

**Safety Checks:**
- ✅ **One at a Time:** Blocks if another feature branch exists
- ✅ **Clean State:** Requires clean working directory for new branch creation
- ✅ **Idempotent:** Safe to call multiple times
  - Already on branch → no-op
  - Branch exists elsewhere → checkout
  - Branch doesn't exist → create

**Error Conditions:**
```python
# Another feature branch exists
"ERROR: Cannot create new feature branch. Existing feature branch(es) detected: feature/task-999-other"

# Working directory dirty
"ERROR: Cannot create new feature branch. Working directory has uncommitted changes"
```

**Test Coverage:** 4 tests (2 failure, 2 success)

---

### 2. `git_add`

**Purpose:** Stage files for commit

**Safety Checks:**
- ✅ **Block on Main:** Cannot stage files on `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot stage files on main branch. You must be on a feature branch"

# On non-feature branch (e.g., develop)
"ERROR: Cannot stage files on branch 'develop'. You must be on a feature branch"
```

**Test Coverage:** 3 tests (2 failure, 1 success)

---

### 3. `git_smart_commit`

**Purpose:** Commit staged files with Protocol 101 v3.0 enforcement

**Safety Checks:**
- ✅ **Block on Main:** Cannot commit to `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch
- ✅ **Staged Files Required:** Verifies files are staged before committing

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot commit directly to main branch. You must be on a feature branch"

# On non-feature branch
"ERROR: Cannot commit on branch 'develop'. You must be on a feature branch"

# No staged files
"ERROR: No files staged for commit. Please use git_add first"
```

**Test Coverage:** 4 tests (3 failure, 1 success)

---

### 4. `git_push_feature`

**Purpose:** Push feature branch to origin

**Safety Checks:**
- ✅ **Block on Main:** Cannot push `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch
- ✅ **Remote Hash Verification:** Verifies remote hash matches local after push

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot push main branch directly. You must be on a feature branch"

# On non-feature branch
"ERROR: Cannot push branch 'develop'. You must be on a feature branch"

# Hash mismatch (WARNING, not blocking)
"WARNING: Push completed but remote hash (abc123de) does not match local (xyz789ab)"
```

**Test Coverage:** 4 tests (2 failure, 2 success)

---

### 5. `git_finish_feature`

**Purpose:** Cleanup after PR merge (delete branches, sync main)

**Safety Checks:**
- ✅ **Block Main:** Cannot finish `main` branch
- ✅ **Feature Branch Only:** Must be `feature/task-XXX-desc` format
- ✅ **Clean State:** Requires clean working directory
- ✅ **Merge Verification:** Verifies branch is merged into `main` before deletion
  - Pulls `main` first to ensure local is up-to-date
  - Checks `git branch --merged main`
  - **Prevents data loss** by blocking unmerged branch deletion

**Error Conditions:**
```python
# Trying to finish main
"ERROR: Cannot finish 'main' branch. It is the protected default branch"

# Invalid branch name
"ERROR: Invalid branch name 'develop'. Can only finish feature branches"

# Working directory dirty
"ERROR: Working directory is not clean. Please commit or stash changes"

# Branch not merged
"ERROR: Branch 'feature/task-123-test' is NOT merged into main. Cannot finish/delete an unmerged feature branch"
```

**Test Coverage:** 5 tests (4 failure, 1 success)

---

## Workflow Enforcement

### Required Sequence

![git_workflow_sequence](../../docs/architecture_diagrams/workflows/git_workflow_sequence.png)

*[Source: git_workflow_sequence.mmd](../../docs/architecture_diagrams/workflows/git_workflow_sequence.mmd)*

### Out-of-Sequence Prevention

| Attempted Action | Without | Result |
|------------------|---------|--------|
| `git_add` | `git_start_feature` | ❌ Blocked: "Cannot stage files on main branch" |
| `git_smart_commit` | `git_add` | ❌ Blocked: "No files staged for commit" |
| `git_push_feature` | `git_smart_commit` | ⚠️ Allowed (git handles "everything up-to-date") |
| `git_finish_feature` | PR Merge | ❌ Blocked: "Branch is NOT merged into main" |

---

## Test Suite

### Location
- **Unit Tests:** `tests/test_git_ops.py` (10 tests)
- **Safety Tests:** `tests/mcp_servers/git_workflow/test_tool_safety.py` (20 tests)
- **Total:** 30 tests, 100% passing ✅

### Coverage Breakdown

```
git_add:           3 tests (2 failure, 1 success)
git_start_feature: 4 tests (2 failure, 2 success)
git_smart_commit:  4 tests (3 failure, 1 success)
git_push_feature:  4 tests (2 failure, 2 success)
git_finish_feature: 5 tests (4 failure, 1 success)
```

### Running Tests

```bash
# All git tests
pytest tests/test_git_ops.py tests/mcp_servers/git_workflow/ -v

# Safety tests only
pytest tests/mcp_servers/git_workflow/test_tool_safety.py -v

# Specific test
pytest tests/mcp_servers/git_workflow/test_tool_safety.py::TestGitToolSafety::test_finish_feature_blocks_unmerged -v
```

---

## Protocol Compliance

### Protocol 101 v3.0: Functional Coherence

**Enforcement:** `git_smart_commit` automatically runs the test suite via pre-commit hook

**Workflow:**
1. User stages files with `git_add`
2. User calls `git_smart_commit` with message
3. Pre-commit hook executes `./scripts/run_genome_tests.sh`
4. If tests pass → commit succeeds
5. If tests fail → commit is blocked

**No Manual Intervention Required** - The hook enforces functional coherence automatically.

---

## Migration from `git_sync_main`

### Why Removed?

**Problem:** Agents were calling `git_sync_main` at inappropriate times:
- Before PR was merged
- While on feature branches
- Without verifying remote state

**Solution:** Removed tool entirely. Sync now happens **only** in `git_finish_feature` after merge verification.

### Migration Path

**Old Workflow:**
```python
git_finish_feature("feature/task-123-test")
git_sync_main()  # Manual sync
```

**New Workflow:**
```python
git_finish_feature("feature/task-123-test")  # Syncs main automatically
```

---

## Future Enhancements

### Recommended Additions

1. **Remote Tracking Verification**
   - Check if remote exists before push
   - Verify network connectivity

2. **Ahead/Behind Check**
   - Warn if remote is ahead before push
   - Suggest pull/rebase

3. **Force Push Warning**
   - Add explicit confirmation for `force=True`
   - Or block force push entirely

4. **Stale Branch Detection**
   - Warn if feature branch is behind main
   - Suggest rebase

---

## Troubleshooting

### Common Errors

**"Cannot stage files on main branch"**
- **Cause:** Attempted `git_add` on `main`
- **Solution:** Run `git_start_feature` first

**"No files staged for commit"**
- **Cause:** Attempted `git_smart_commit` without staging
- **Solution:** Run `git_add` first

**"Branch is NOT merged into main"**
- **Cause:** Attempted `git_finish_feature` before PR merge
- **Solution:** Merge PR on GitHub first, then retry

**"Existing feature branch(es) detected"**
- **Cause:** Attempted to create second feature branch
- **Solution:** Finish current feature branch first with `git_finish_feature`

---

## Related Documentation

- [Git Workflow MCP README](README.md)
- [MCP Operations Inventory](../../docs/operations/mcp/mcp_operations_inventory.md)
- [Protocol 101 v3.0](../../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md)

---

**Maintainer:** Project Sanctuary Team  
**Status:** Production Ready ✅
