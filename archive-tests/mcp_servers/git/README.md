# Git Server Test Suite

## Overview
This directory contains the comprehensive test suite for the `mcp_servers.git` module, ensuring compliance with **Protocol 101 v3.0** (Unbreakable Commit) and **Protocol 122** (Poka-Yoke).

## Structure

```
tests/mcp_servers/git/
├── conftest.py              # Shared fixtures (GitOperations mock, repo setup)
├── e2e/                     # End-to-End Tests
│   └── test_git_e2e.py      # Full system verification (read/write cycles)
├── integration/             # Integration Tests
│   └── test_operations.py   # GitOperations class integration
├── unit/                    # Unit Tests
│   └── test_validator.py    # Validator logic checks
├── test_squash_merge.py     # Squash merge workflow verification
└── test_tool_safety.py      # Safety enforcement (Poka-Yoke) tests
```

## Running Tests

Run the full suite from the project root:

```bash
pytest tests/mcp_servers/git/ -v
```

### Specific Categories

**Safety Checks Only:**
```bash
pytest tests/mcp_servers/git/test_tool_safety.py -v
```

**Integration Tests:**
```bash
pytest tests/mcp_servers/git/integration/ -v
```

## Key Test Cases

### Safety (`test_tool_safety.py`)
- **Main Protection:** Verifies `git_add`, `commit`, `push` are blocked on `main`.
- **Dirty State:** specific checks for uncommitted changes before sensitive ops.
- **Workflow:** Ensures `start_feature` -> `finish_feature` lifecycle.

### Squash Merge (`test_squash_merge.py`)
- Simulates a GitHub squash merge scenario where local history diverges from remote.
- Verifies that `finish_feature` correctly detects the merge via `git merge-base --is-ancestor` logic instead of relying solely on commit hashes.

## Requirements
- `pytest`
- `pytest-asyncio`
- Git installed on the system

### Layer 2: Integration Tests (ISOLATED)
- **Location:** `integration/`
- **Focus:** Real git operations using **TEMPORARY REPOSITORIES**.
- **Strategy:**
  - Tests verify `GitOperations` class methods.
  - A `temp_repo` fixture creates a **clean, isolated git repository** in `/tmp` for *every* test function.
  - Tests execute git commands (init, add, commit, branch) against this temp repo.
  - **SAFETY:** These tests **NEVER** touch the real Project Sanctuary repository or its branches.
- **Run:** `pytest tests/mcp_servers/git/integration/`

### Layer 3: E2E Tests
- **Location:** `e2e/`
- **Focus:** End-to-end user workflows.

## Key Test Files

| File | Purpose |
|------|---------|
| `unit/test_validator.py` | Unit validation of input parameters and safety rules. |
| `integration/test_operations.py` | **MAIN SUITE**: Tests every Git operation in isolation (add, commit, branch, status, log). |

## Integration Test Safety
> **Crucial:** The integration tests use `tempfile.mkdtemp()` to create disposable git repositories.
> This ensures that running `pytest` does not modify your actual working directory, change branches, or create conflicting git state.
