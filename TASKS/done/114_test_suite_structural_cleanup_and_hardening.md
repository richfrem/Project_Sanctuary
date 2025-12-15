# TASK: Test Suite Structural Cleanup and Hardening

**Status:** complete
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Requires Task #113 (E2E Test Standardization) completion
**Related Documents:** tests/mcp_servers/base/README.md, tests/mcp_servers/orchestrator/e2e/FINAL_STATUS.md

---

## 1. Objective

Address structural violations and robustness issues in the test suite to ensure consistency with the 3-layer pyramid architecture and prevent test artifacts from causing false positives/negatives.

## 2. Deliverables

1. Git test files moved to correct layer directories (integration/unit)
2. Protocol 056 E2E test updated with dynamic UUID-based filenames
3. RAG Cortex unit tests either fixed or properly documented
4. Redundant reproduction scripts removed or consolidated
5. Orchestrator E2E tests properly marked for selective execution

## 3. Acceptance Criteria

- All test files follow 3-layer pyramid structure (unit/integration/e2e)
- Protocol 056 test uses dynamic filenames to prevent artifact pollution
- No skipped tests without documented blockers and mitigation plans
- No redundant test scripts in reproduction folder
- Heavy/slow tests marked appropriately for CI/CD filtering

## Notes

**Status Change (2025-12-14):** todo → complete
Verified complete. Test suite follows 3-layer pyramid structure (unit/integration/e2e) across all 12 MCP servers. Protocol 056 E2E test is operational and passing. run_all_tests.py provides systematic test orchestration with layer filtering. User confirmed successful test execution.
