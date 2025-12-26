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
- [ ] Audit all E2E tests creating files (filesystem, domain/chronicle/adr, git).
- [ ] Identify the specific filenames/patterns used.
- [ ] Add `pytest` yield fixtures to delete these specific files after the test completes.
- [ ] Verify cleanup works on both success and failure.
