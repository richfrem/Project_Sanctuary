# TASK: Establish Robust MCP E2E Testing Framework

**Status:** backlog
**Priority:** High
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** TASKS/108_E2E_Testing_Plan.md

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
# Plan: Establish Robust MCP E2E Testing Framework (Task 108)

**Objective:** Enable true End-to-End (E2E) testing by running the actual MCP server fleet and validating behavior via the MCP Protocol.

## 1. Current State Assessment
- **Unit/Integration Layers:** Robust and standardized.
- **E2E Layer:** Currently consists of "Integration+" tests (direct Python method calls to `Operations` classes) rather than true protocol tests.
- **Infrastructure:**
  - `mcp_servers/start_mcp_servers.py`: Manages server startup but is interactive/human-centric.
  - `tests/run_all_tests.py`: Has placeholders for E2E but no harness.
- **Gap:** Missing a mechanism to:
  1. Boot the full server fleet in a headless test environment.
  2. Send MCP protocol requests (JSON-RPC) to specific servers.
  3. Validate responses across server boundaries.

## 2. Technical Architecture

### A. The Test Harness (`BaseE2ETest`)
We need a `BaseE2ETest` class (in `tests/mcp_servers/base/`) that:
- Uses `pytest` fixtures with `scope="session"` to boot the fleet once per test run.
- Wraps `start_mcp_servers.py` using `subprocess.Popen`.
- Implements (or uses) a lightweight **MCP Client** to communicate with the stdio streams of the subprocesses.

### B. The Client (`TestMCPClient`)
Since our servers run as subprocesses of `start_mcp_servers.py` (or directly), the test harness needs to attach to their Input/Output streams.
*Challenge:* `start_mcp_servers.py` aggregates output. We might need to spawn servers individually in the test harness OR modify `start_mcp_servers.py` to expose individual pipes (harder).
*Better Approach:* The E2E harness should use the **same config** (`mcp_config.json`) to spawn each server process individually using `fastmcp run` or `python server.py`, maintaining a handle to each.

### C. Test Scenarios (Critical Paths)
1. **Knowledge Loop:** Create Chronicle Entry -> Verify Cortex Ingestion (via `get_stats`).
2. **Task Execution:** Create Task (Task MCP) -> Dispatch Agent (Council) -> Verify Agent sees Task.
3. **Strategic Loop:** Protocol 056 (Orchestrator) -> Full system cycle.

## 3. Implementation Plan

### Phase 1: Harness Development
1. **Refactor `start_mcp_servers.py`** to allow module-level import of the "server starter" logic, so tests can invoke it programmatically.
2. **Create `tests/mcp_servers/base/mcp_test_client.py`**: A helper class that spawns an MCP server process and speaks JSON-RPC over stdio.
3. **Create `BaseE2ETest`**: Provides a dictionary of `server_name -> client` instances.

### Phase 2: Pilot Implementation (Chronicle + Cortex)
1. Create `tests/mcp_servers/chronicle/e2e/test_chronicle_cortex_flow.py`.
2. Test:
   - Call `chronicle_create_entry` (via Client).
   - Wait 5s.
   - Call `cortex_query` (via Client).
   - Assert entry content exists in RAG.

### Phase 3: Rollout & Integration
1. Extend to all 12 servers.
2. Update `tests/run_all_tests.py` to execute the E2E suite using the new harness.
3. Add to CI pipeline (with Docker/Podman requirements).

## 4. Work Items (Task 108 Breakdown)
- [ ] Create `mcp_test_client.py` (JSON-RPC wrapper).
- [ ] Create `BaseE2ETest` fixture logic.
- [ ] Implement "Chronicle -> Cortex" E2E Test.
- [ ] Implement "Council -> Tool" E2E Test.
- [ ] **Port "Protocol 056" Test (`test_strategic_crucible.py`) to use new E2E Harness.**
- [ ] Integrate into `tests/run_all_tests.py` --layer e2e.
