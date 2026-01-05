# Git Workflow & File Retention Policy

**Purpose:** Prevent accidental file deletions and enforce proper git workflow.

## Git Workflow Rules (MANDATORY)

### 1. Feature Branch Requirement
**NEVER commit directly to `main`.**

```bash
# ✅ CORRECT: Create feature branch first
git checkout -b feature/task-XXX-description
# Make changes
git add specific-files.py
git commit -m "feat: description"
git push origin feature/task-XXX-description
# Create PR on GitHub

# ❌ WRONG: Committing to main
git checkout main
git commit -m "changes"  # VIOLATION
```

### 2. Explicit File Staging
**NEVER use `git add .` or `git add --all` without review.**

```bash
# ✅ CORRECT: Stage specific files
git add path/to/file1.py
git add path/to/file2.md
git diff --cached  # ALWAYS review before commit

# ❌ WRONG: Blind staging
git add .  # Can stage deletions accidentally
git add --all  # Can stage deletions accidentally
```

### 3. Pre-Commit Review
**ALWAYS review staged changes before committing.**

```bash
# Required before every commit
git diff --cached
git status

# Verify no unexpected deletions
git diff --cached --name-status | grep "^D"
```

## File Retention Policy

### Protected Directories (NEVER DELETE without explicit approval)

1. **`council_orchestrator/`** - Full orchestrator implementation
   - Contains all working code
   - Required by MCP server
   - Deletion requires explicit user approval

2. **`01_PROTOCOLS/`** - Canonical protocols
   - Protected by git safety rules
   - Never delete protocol files

3. **`.agent/`** - Agent configuration
   - Protected by git safety rules
   - Critical for AI workflows

4. **`tasks/done/`** - Completed tasks
   - Archive of finished work
   - Keep for historical reference

### Cleanup-Allowed Directories (with caution)

1. **`WORK_IN_PROGRESS/`**
   - Can clean up old experiments
   - Must verify no active work
   - Create archive before deletion

2. **`tasks/backlog/`**
   - Can move to `done/` or archive
   - Don't delete, relocate

3. **Temporary files**
   - `*.pyc`, `__pycache__/`, `.DS_Store`
   - Safe to delete (in .gitignore)

## MCP Git Tools - Correct Usage

### Using git_workflow MCP

```python
# Step 1: ALWAYS start with feature branch
mcp3_git_start_feature(task_id="022", description="documentation")

# Step 2: Stage specific files
mcp3_git_add(files=["README.md", "docs/guide.md"])

# Step 3: Review (manual)
# Run: git diff --cached

# Step 4: Commit
mcp3_git_smart_commit(message="docs: update README")

# Step 5: Push feature branch
mcp3_git_push_feature()

# Step 6: Create PR (manual or via mcp3_git_create_pr)
```

### NEVER Skip Steps

❌ **Wrong - skipping feature branch:**
```python
mcp3_git_smart_commit(message="changes")  # Commits to main!
```

✅ **Correct - full workflow:**
```python
mcp3_git_start_feature(...)
mcp3_git_add(files=[...])
# Review manually
mcp3_git_smart_commit(...)
mcp3_git_push_feature()
```

## Recovery Procedures

### If Files Are Accidentally Deleted

1. **Don't panic** - files exist in git history
2. **Don't commit** - if not yet committed, restore immediately
3. **Check git status** - see what's staged

```bash
# If files deleted but not committed
git restore path/to/deleted/file

# If files deleted and committed (but not pushed)
git reset --soft HEAD~1  # Undo commit, keep changes
git restore path/to/deleted/file

# If pushed to GitHub
git restore --source=origin/main path/to/deleted/file
```

### Restoration from GitHub

```bash
# Restore entire directory from GitHub
git restore --source=origin/main directory/

# Restore specific file
git restore --source=origin/main path/to/file
```

## Branch Protection (GitHub Settings)

### Required Settings

1. **Require pull request before merging** ✅
2. **Require approvals: 1** ✅
3. **Dismiss stale reviews** ✅
4. **Require status checks** (if CI/CD configured)
5. **Include administrators** ✅ (enforce for everyone)

### How to Verify

```bash
# Check if branch protection is active (requires gh CLI auth)
gh api repos/richfrem/Project_Sanctuary/branches/main/protection
```

## Enforcement Checklist

Before ANY git operation:
- [ ] Am I on a feature branch? (`git branch` shows `feature/...`)
- [ ] Have I staged only specific files? (`git diff --cached`)
- [ ] Have I reviewed the diff? (No unexpected deletions)
- [ ] Is this commit necessary? (Not committing just to commit)

## Consequences of Violations

**Committing to main:**
- Violates Protocol 101
- Bypasses code review
- Risk of accidental deletions
- Must be reverted and redone on feature branch

**Using `git add .`:**
- Can stage unintended deletions
- Can stage sensitive files
- Can stage incomplete work
- Always use explicit file lists

**Skipping diff review:**
- Highest risk of accidental deletions
- Can commit broken code
- Can commit secrets
- NEVER skip this step

## Summary

1. ✅ **Always use feature branches**
2. ✅ **Always stage specific files**
3. ✅ **Always review diffs before commit**
4. ✅ **Never delete from protected directories**
5. ✅ **Use MCP git tools correctly**
6. ✅ **Verify branch protection is active**
