# Git Operations Guide (Project Sanctuary)

**Governing Doctrine:** Protocol 101 (The Unbreakable Commit) & ADR 074 (Git Robustness).

## 1. The Golden Rule
**NO DIRECT WORK ON MAIN.**
Always create a feature branch (`feat/name` or `fix/name`) before editing files.

## 2. MCP Tools Overview
The `sanctuary_git` server provides "Smart" tools that handle safety checks for you.

| Tool | Purpose | "Happy Path" Behavior |
|------|---------|------------------------|
| `git_start_feature` | Creating new branches | Auto-fetches `origin/main` to ensure you branch off fresh code. |
| `git_smart_commit` | Staging & Committing | Runs Poka-Yoke tests. If docs-only, skips heavy tests. |
| `git_finish_feature` | Merging & Cleaning | Checks `origin/main` to confirm merge before deleting local branch. |

## 3. Standard Workflow

### Step 1: Start
```python
git_start_feature(branch_name="feat/new-docs")
```

### Step 2: Work & Commit
```python
# Edit files...
# Then commit (automatically stages all changes if you don't specify files)
git_smart_commit(message="feat: add new documentation")
```

### Step 3: Push
```python
git_push_feature()
```

### Step 4: Finish (After PR Merge)
Once the user merges the PR on GitHub:
```python
git_finish_feature()
```
*Note: This will fail safely if the PR isn't actually merged on remote.*

## 4. Troubleshooting

### "Non-Fast-Forward" Error
1.  Run `git fetch origin`.
2.  If on feature branch: `git merge origin/main` (to update).
3.  If on main: `git reset --hard origin/main` (only if you have no local changes to save).

### "Poka-Yoke Failed"
Read the error log.
- **Lint Error**: Run `ruff check .` to fix formatting.
- **Test Error**: Run `pytest` to fix logic.
- **Docs Update**: Ensure you created `implementation_plan.md` (if required).
