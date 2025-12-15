# TASK: Verify All MCP Servers E2E Tests

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Run and verify E2E tests for all 12 MCP servers, ensuring consistency in implementation and invocation via run_all_tests.py.

## 2. Deliverables

1. Updated .gitignore
2. Verification Report
3. Consistent E2E test structure

## 3. Acceptance Criteria

- All 12 MCP servers have E2E test directories.
- run_all_tests.py correctly discovers and runs E2E tests for all 12 servers.
- operations_instrumentation.json is ignored by git.
- E2E tests pass or are explicitly skipped with valid reasons.
