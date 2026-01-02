# TASK: Implement Comprehensive Gateway MCP E2E Test Suite

**Status:** in-progress
**Priority:** High
**Lead:** Antigravity AI Agent
**Dependencies:** None
**Related Documents:** 
- [tests/mcp_servers/gateway/README.md](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/tests/mcp_servers/gateway/README.md) (Test Pyramid Architecture)
- [ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md)
- [GATEWAY_VERIFICATION_MATRIX.md](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/docs/architecture/mcp_servers/gateway/operations/GATEWAY_VERIFICATION_MATRIX.md)
- [mcp_servers/gateway/fleet_registry.json](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/mcp_servers/gateway/fleet_registry.json)
- tasks/146_fix_issues_with_gateway_mcp_operations.md

---

## 1. Objective

Create and execute a systematic, verifiable test suite for all 87 Gateway MCP operations following the **4-Tier Zero-Trust Pyramid** (see tests/mcp_servers/gateway/README.md) with detailed execution logging.

## 2. Test Structure (Per Cluster)

Following the established pattern in `tests/mcp_servers/gateway/`:

```
tests/mcp_servers/gateway/clusters/
├── sanctuary_cortex/
│   ├── unit/           # Tier 1: Internal Logic
│   ├── integration/    # Tier 2: SSE Server Readiness
│   └── e2e/            # Tier 3: Gateway RPC (The Bridge)
├── sanctuary_domain/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── sanctuary_filesystem/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── sanctuary_git/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── sanctuary_network/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── sanctuary_utils/
    ├── unit/
    ├── integration/
    └── e2e/
```

## 3. Deliverables

1. **Tier 1 (Unit):** Complete test coverage in `tests/mcp_servers/gateway/clusters/<cluster>/unit/`
2. **Tier 2 (Integration):** SSE handshake and tool validation in `tests/mcp_servers/gateway/clusters/<cluster>/integration/`
3. **Tier 3 (E2E):** Full Gateway RPC tests in `tests/mcp_servers/gateway/clusters/<cluster>/e2e/`
4. Detailed execution log following Task 146 pattern with timestamps, inputs, outputs, and durations
5. Learning package protection: Use `tests/fixtures/test_docs/` only — NEVER modify `.agent/learning/`
6. Updated GATEWAY_VERIFICATION_MATRIX.md with evidence-backed checkmarks

## 4. Acceptance Criteria

- [ ] All 6 clusters have unit, integration, and e2e test directories
- [ ] Tier 1 (Unit): 100% operations.py logic coverage
- [ ] Tier 2 (Integration): Health + SSE handshake tests passing for all 6 clusters
- [ ] Tier 3 (E2E): Gateway RPC tests for all 87 tools (via bridge.py simulation)
- [ ] `test_cortex_ingest_full` shows 2-5 minute execution with progress logging every 15s
- [ ] `.agent/learning/` directory unchanged after test run (verified by hash comparison)
- [ ] Minimum 85/87 tests passing (97% pass rate per current GATEWAY_VERIFICATION_MATRIX)
- [ ] GATEWAY_VERIFICATION_MATRIX.md updated ONLY from test execution results

## 5. Current Progress

| Cluster | Unit | Integration | E2E |
|---------|:----:|:-----------:|:---:|
| sanctuary_utils | ✅ | ✅ | ✅ |
| sanctuary_filesystem | ✅ | ✅ | ✅ |
| sanctuary_network | ✅ | ✅ | ✅ |
| sanctuary_git | ✅ | ⚠️ | ⚠️ |
| sanctuary_cortex | ✅ | ✅ | ✅ |
| sanctuary_domain | ✅ | ✅ | ✅ |

> **⚠️ Git:** Gateway RPC timeout (SSL handshake issue, not tool logic)

## Notes

- **Test Pyramid Reference:** See `tests/mcp_servers/gateway/README.md` for tier definitions
- **Diagnosis Rule:** 
  - If Integration test fails → **Code bug** in cluster
  - If only E2E test fails → **Network/infrastructure** timeout
- **CRITICAL:** Use `tests/fixtures/test_docs/` for ingestion tests — NEVER `.agent/learning/`
