# TASK: Establish Robust MCP E2E Testing Framework

**Status:** complete
**Priority:** High
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** tasks/108_E2E_Testing_Plan.md

---

## 1. Objective

Establish a robust E2E testing framework that validates the entire Sanctuary system by running actual MCP servers and communicating via the MCP protocol.

## 2. Deliverables

1. BaseE2ETest implementation in tests/mcp_servers/base/base_e2e_test.py
2. E2E Test Suite for Critical Flows
3. Documentation on running E2E tests

## 3. Acceptance Criteria

- BaseE2ETest can start/stop full MCP server fleet via start_mcp_servers.py
- At least 3 critical cross-server workflows are tested E2E (e.g. Chronicle->Cortex, Council->Agent)
- E2E tests run via `python3 tests/run_all_tests.py --layer e2e`
- CI/CD check runs E2E tests successfully

## Notes

**Status Change (2025-12-14):** backlog → complete
E2E framework established, tests passing, polyglot ingestion added.
