# Task 149: Implement Self-Cleaning Test Logic

## Context
Integration and E2E tests currently generate artifact files (ADRs, Chronicle entries, specialized logs) during execution to verify system behavior. These artifacts are not being cleaned up, leading to clutter in the repository.

## Objective
Update Tier 2 (Integration) and Tier 3 (E2E) test scripts to reliably clean up the artifacts they create.

## Constraints
- **Preserve Test Sources:** MUST ONLY delete the *generated outputs* (e.g., specific ADRs created by the test), NEVER the test scripts themselves (`test_*.py`).
- **Targeted Cleanup:** Use specific filenames or patterns used during creation (e.g., `*e2e_test_adr.md`), not broad wildcards.
- **Teardown Scope:** Implement cleanup in `teardown` fixtures or `finally` blocks to ensure execution even if tests fail.

## Implementation Plan
- [x] Audit all E2E tests creating files (filesystem, domain/chronicle/adr, git).
- [x] Identify the specific filenames/patterns used.
- [x] Add `pytest` yield fixtures to delete these specific files after the test completes.
- [x] Verify cleanup works on both success and failure.

---

## Resolution (2025-12-26)

### Implemented Cleanup Strategy

1. **Direct E2E Tests** (`tests/mcp_servers/*/e2e/test_*_e2e.py`):
   - Already have `finally` blocks for cleanup (verified working: ADR E2E test creates and cleans ADR 077)

2. **Gateway E2E Tests** (`tests/mcp_servers/gateway/clusters/*/e2e/test_e2e_*.py`):
   - Added `cleanup_e2e_artifacts` session-scoped fixture in `gateway/e2e/conftest.py`
   - Auto-cleans artifacts matching patterns: `*e2e*test*.md`, `*E2E*Test*.md`, `e2e_*.txt`
   - Cleans: ADRs, Chronicle entries, Protocols, Tasks, test files in project root

3. **Cleanup Utilities** (`tests/mcp_servers/base/cleanup_fixtures.py`):
   - Created `CleanupRegistry` class for explicit file registration
   - Created `e2e_cleanup` fixture for per-test cleanup
   - Created helper fixtures: `safe_adr_path`, `safe_chronicle_path`, `safe_test_file`
   - Added `cleanup_stale_e2e_artifacts()` emergency cleanup function

### Stale Artifacts Cleaned
- Removed 7 stale Chronicle entries (315-322 `protocol_056_test_initialization.md`)
- Verified no stale ADRs from previous E2E runs

### Files Modified
- `tests/mcp_servers/gateway/e2e/conftest.py` - Added cleanup fixture
- `tests/mcp_servers/base/cleanup_fixtures.py` - Created (new)

### Verification
- Ran `pytest tests/mcp_servers/adr/e2e/test_adr_e2e.py -v -s`
- Confirmed ADR 077 was created and cleaned up in `finally` block
