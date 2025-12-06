# Git MCP Testing

This directory contains the testing hierarchy for the Git MCP server, in accordance with **ADR 048**.

## Testing Layers

### Layer 1: Unit Tests
- **Location:** `unit/`
- **Focus:** Isolated logic, path validation, safety checks.
- **Run:** `pytest tests/mcp_servers/git/unit/`

### Layer 2: Integration Tests
- **Location:** `integration/`
- **Focus:** Real git operations using temporary repositories.
- **Key Features:**
  - `git_roots` fixture: Creates a simulated remote/local environment.
  - Test coverage for `finish_feature` (merge checks, squash detection, cleanup).
- **Run:** `pytest tests/mcp_servers/git/integration/`

### Layer 3: MCP Operations
- **Location:** N/A (Manual verification via MCP Client or `server.py` inspection)
- **Focus:** End-to-end tool execution.
- **Status:** Verified via integration tests covering the core logic refactored into `GitOperations`.

## Key Test Files

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures (`git_roots`, `git_ops_mock`). |
| `unit/test_validator.py` | Unit validation tests. |
| `integration/test_operations.py` | Comprehensive integration tests for all Git tools. |
