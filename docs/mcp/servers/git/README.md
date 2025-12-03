# Git MCP Server Documentation

## Overview

Git MCP provides safe, validated Git operations for Project Sanctuary. It includes comprehensive safety checks, branch management, and smart commit capabilities.

## Key Concepts

- **Safety Checks:** Pre-commit validation (no secrets, proper formatting)
- **Branch Management:** Feature branches, squash merges, protected branches
- **Smart Commits:** Conventional commit format with automatic validation
- **Dry Run Mode:** Preview operations before execution

## Server Implementation

- **Server Code:** [mcp_servers/git/server.py](../../../mcp_servers/git/server.py)
- **Operations:** [mcp_servers/git/git_ops.py](../../../mcp_servers/git/git_ops.py)
- **Safety:** [mcp_servers/git/safety.py](../../../mcp_servers/git/safety.py)

## Testing

- **Test Suite:** [tests/mcp_servers/git/](../../../tests/mcp_servers/git/)
- **Status:** ✅ 20/20 tests passing
- **Coverage:** Comprehensive safety checks and workflow tests

## Operations

### `git_status`
Get current repository status

### `git_add`
Stage files for commit

### `git_smart_commit`
Create a commit with conventional commit format and validation

**Example:**
```python
git_smart_commit(
    message="feat(mcp): Add Council vs Orchestrator documentation",
    dry_run=False
)
```

### `git_create_feature_branch`
Create a new feature branch

### `git_finish_feature`
Finish a feature branch (squash merge to main)

### `git_push_feature`
Push feature branch to remote

### `git_list_branches`
List all branches

## Safety Features

- **Secret Detection:** Prevents committing API keys, tokens, passwords
- **File Size Limits:** Prevents committing large files
- **Protected Branches:** Cannot directly commit to main/master
- **Conventional Commits:** Enforces commit message format

## Status

✅ **Fully Operational** - All safety checks and operations tested
