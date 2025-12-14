# Git MCP Testing

This directory contains the testing hierarchy for the Git MCP server, in accordance with **ADR 048** and **ADR 055**.

## Testing Layers

### Layer 1: Unit Tests
- **Location:** `unit/`
- **Focus:** Isolated logic, path validation, safety checks.
- **Run:** `pytest tests/mcp_servers/git/unit/`

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
