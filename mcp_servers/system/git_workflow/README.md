# Git Workflow MCP Server

**Domain:** `project_sanctuary.system.git_workflow`  
**Version:** 2.0.0  
**Status:** Production Ready (Protocol 101 v3.0 Compliant)

---

## Overview

The Git Workflow MCP server provides **Protocol 101 v3.0-compliant git operations** with an opinionated, safe workflow for feature development. It enforces **Functional Coherence** through automated test suite execution and integrates with GitHub for PR creation.

**Key Principle:** Safety-first, test-driven git operations with functional integrity verification.

---

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Git** installed and configured
3. **GitHub CLI** (optional, for PR creation): `brew install gh && gh auth login`

### Start the MCP Server

**Local Development:**
```bash
cd /Users/richardfremmerlid/Projects/Project_Sanctuary
python3 -m mcp_servers.system.git_workflow.server
```

**Via Claude Desktop:**
Already configured in `claude_desktop_config.json`. Just restart Claude.

---

## Tools (9)

### 1. Workflow Tools

#### `git_start_feature(task_id, description)`
Create and checkout a new feature branch.

**Example:**
```python
git_start_feature("045", "smart-git-mcp")
# Creates: feature/task-045-smart-git-mcp
```

#### `git_smart_commit(message)` ⭐
**The "Smart" Part:** Automatically enforces Protocol 101 v3.0 (Functional Coherence).

**What it does:**
1. Commits staged files
2. Pre-commit hook automatically runs `./scripts/run_genome_tests.sh`
3. All tests must pass for commit to proceed
4. Ensures functional integrity before accepting commit

**Example:**
```python
git_smart_commit("Implement feature X")
# Result: Commit only if all tests pass
```

#### `git_push_feature()`
Push current feature branch to origin.

#### `git_create_pr(title, body, base)` ⭐
Create a GitHub Pull Request using GitHub CLI.

**Example:**
```python
git_create_pr(
    title="Add Smart Git MCP",
    body="Implements Protocol 101 compliance",
    base="main"
)
# Returns: PR URL
```

#### `git_finish_feature(branch_name)`
Cleanup after PR is merged (on GitHub).

**What it does:**
1. Checkout main
2. Pull latest
3. Delete local feature branch

#### `git_sync_main()`
Pull latest changes from origin/main.

---

### 2. Read-Only Tools

#### `git_get_status()`
Get repository status (branch, staged, modified, untracked files).

#### `git_diff(cached, file_path)`
Show changes in working directory or staged files.

**Examples:**
```python
git_diff(cached=False)  # Unstaged changes
git_diff(cached=True)   # Staged changes
git_diff(file_path="core/git/git_ops.py")  # Specific file
```

#### `git_log(max_count, oneline)`
Show commit history.

**Examples:**
```python
git_log(max_count=10, oneline=False)  # Last 10 commits (detailed)
git_log(max_count=5, oneline=True)    # Last 5 commits (compact)
```

---

## Complete Workflow

```
1. git_start_feature("046", "configure-mcp-client")
   → Creates feature/task-046-configure-mcp-client

2. (Make your changes)

3. git_diff(cached=False)
   → Review unstaged changes

4. (Stage files: git add ...)

5. git_diff(cached=True)
   → Review staged changes

6. git_smart_commit("Add MCP client configuration")
   → Commit (tests run automatically via pre-commit hook)

7. git_push_feature()
   → Push to GitHub

8. git_create_pr("Configure MCP Client", "Adds .agent/mcp_config.json")
   → Create PR on GitHub

9. (Review and merge PR on GitHub)

10. git_finish_feature("feature/task-046-configure-mcp-client")
    → Cleanup local branch
```

---

## Security Features

### 1. Base Directory Restriction
Set `GIT_BASE_DIR` environment variable to restrict all operations to a specific directory tree.

```bash
export GIT_BASE_DIR=/Users/richardfremmerlid/Projects/Project_Sanctuary
```

### 2. Path Sanitization
All file paths are validated and sanitized to prevent directory traversal attacks.

### 3. No Destructive Operations
The following dangerous operations are **NOT** exposed:
- `git reset --hard`
- `git rebase`
- `git push --force`
- Branch deletion (except via `finish_feature`)

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REPO_PATH` | Repository root path | `.` (current directory) |
| `GIT_BASE_DIR` | Security sandbox (optional) | None |
| `PROJECT_ROOT` | Project root for PYTHONPATH | Required |

### Claude Desktop Config

```json
{
  "mcpServers": {
    "git_workflow": {
      "displayName": "Git Workflow MCP",
      "command": "/usr/local/bin/python3",
      "args": ["-m", "mcp_servers.system.git_workflow.server"],
      "env": {
        "PYTHONPATH": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "GIT_BASE_DIR": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      },
      "cwd": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    }
  }
}
```

---

## Comparison to Other Git MCPs

See [git_mcp_comparison.md](file:///Users/richardfremmerlid/.gemini/antigravity/brain/8e7a3729-cc05-40ae-a5dd-38935c512229/git_mcp_comparison.md) for a detailed comparison with cyanheads/git-mcp-server.

**Our Unique Features:**
- ⭐ Automatic Protocol 101 v3.0 enforcement (Functional Coherence)
- ⭐ GitHub PR creation via `gh` CLI
- ⭐ Opinionated, safe workflow (prevents mistakes)

---

## Testing

```bash
# Run all tests
PYTHONPATH=. python3 tests/test_git_ops.py -v

# Test specific functionality
PYTHONPATH=. python3 tests/test_git_ops.py -v TestGitOperations.test_branch_operations
```

---

## Troubleshooting

### GitHub CLI Not Found
```
Error: GitHub CLI (gh) not found
```
**Solution:** Install GitHub CLI: `brew install gh && gh auth login`

### Base Directory Violation
```
Error: Repository path is outside base directory
```
**Solution:** Ensure `REPO_PATH` is within `GIT_BASE_DIR`.

### Test Suite Failed
```
Error: Commit rejected - tests failed
```
**Solution:** Fix failing tests before committing. Run `./scripts/run_genome_tests.sh` to see failures.

---

## Related Documentation

- [Protocol 101 v3.0: The Doctrine of Absolute Stability](../../../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md)
- [Protocol 102 v2.0: The Doctrine of Mnemonic Synchronization](../../../01_PROTOCOLS/102_The_Doctrine_of_Mnemonic_Synchronization.md)
- [ADR 037: MCP Git Strategy - Immediate Compliance](../../../ADRs/037_mcp_git_migration_strategy.md)
- [ADR 019: Cognitive Genome Publishing Architecture](../../../ADRs/019_protocol_101_unbreakable_commit.md)

---

**Last Updated:** 2025-11-29  
**Maintainer:** Project Sanctuary Team
