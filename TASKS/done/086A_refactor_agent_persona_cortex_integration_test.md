# Task 086A: Refactor & Re-enable Agent Persona/Cortex Integration Test

**Status:** TODO  
**Priority:** CRITICAL 🔴  
**Created:** 2025-12-01  
**Estimated Effort:** 2-3 hours

## Objective

Refactor and re-enable the disabled integration test to validate Agent Persona MCP ↔ Cortex MCP communication in the new MCP architecture.

## Background

The original integration test (`tests/integration/test_council_orchestrator_with_cortex.py.disabled`) tested the legacy `council_orchestrator` → `mnemonic_cortex` flow. With the migration to Agent Persona MCP and Cortex MCP, this test needs complete refactoring.

## Current State

**File:** `tests/integration/test_council_orchestrator_with_cortex.py.disabled`

**Issues:**
- ❌ Imports from deleted `council_orchestrator` module
- ❌ Tests old `CortexManager` interface  
- ❌ Mocks chromadb directly instead of using MCP tools
- ❌ Disabled (`.disabled` suffix)

## Target State

**New File:** `tests/integration/test_agent_persona_with_cortex.py` ✅ CREATED

**Status:**
- ✅ File created with 4 comprehensive integration tests
- ⚠️ Tests currently deselected by pytest config
- ⏭️ Need to enable in pytest.ini

## tasks

### Phase 1: Enable Tests ✅ DONE
- [x] Create new integration test file
- [x] Implement test_persona_queries_cortex_for_context
- [x] Implement test_council_dispatch_full_flow
- [x] Implement test_multi_agent_deliberation_with_context
- [x] Implement test_cortex_query_returns_results

### Phase 2: Enable in pytest.ini
- [ ] Update `pytest.ini` to include integration tests
- [ ] Add marker configuration for `@pytest.mark.integration`
- [ ] Run tests to verify they pass

### Phase 3: Cleanup
- [ ] Delete old `.disabled` file
- [ ] Update test documentation
- [ ] Add to CI/CD pipeline (if applicable)

## Test Coverage

### Test 1: `test_persona_queries_cortex_for_context`
**Validates:** Agent Persona MCP can query Cortex MCP for context

**Flow:**
```
persona_dispatch → cortex.query → returns context → agent uses context
```

### Test 2: `test_council_dispatch_full_flow`
**Validates:** Full Council → Agent Persona → Cortex flow

**Flow:**
```
council_dispatch → persona_dispatch → cortex.query → results flow back
```

### Test 3: `test_multi_agent_deliberation_with_context`
**Validates:** Multi-round, multi-agent deliberation with Cortex context

**Flow:**
```
council_dispatch (no agent specified)
  → Round 1: coordinator, strategist, auditor (all query Cortex)
  → Round 2: agents critique each other
  → Final synthesis
```

**Assertions:**
- 3 agents execute
- 2 rounds × 3 agents = 6 packets
- Cortex queried once (at start)
- Final synthesis includes all perspectives

### Test 4: `test_cortex_query_returns_results`
**Validates:** Cortex MCP query operation works correctly

**Flow:**
```
cortex_ops.query → ChromaDB → returns results
```

## Mock Strategy

**Current Implementation:** Mock at MCP tool level
- Mock `get_llm_client` for agent responses
- Mock `CortexOperations.query` for context retrieval
- Mock `chromadb` for Cortex internal operations

**Benefits:**
- Tests actual Council → Agent Persona flow
- Validates MCP interface contracts
- Fast execution (no real DB needed)

## Success Criteria

- [ ] All 4 integration tests pass
- [ ] Tests run in CI/CD pipeline
- [ ] Old `.disabled` file removed
- [ ] Documentation updated
- [ ] Code coverage > 80% for integration paths

## Dependencies

- ✅ Agent Persona MCP implementation
- ✅ Cortex MCP implementation
- ✅ Council MCP implementation
- ⏭️ pytest.ini configuration update

## Related tasks

- Task #077: Council MCP Server (DONE)
- Task #078: Agent Persona MCP & Orchestrator Refactoring (DONE)
- Task #086: Post-Migration Validation (IN PROGRESS)

## Notes

### Architecture Verification ✅
The multi-agent deliberation logic is correctly implemented in `mcp_servers/lib/council/council_ops.py`:
- Lines 135-173: Multi-round deliberation loop
- Line 143: Calls Agent Persona MCP via `persona_ops.dispatch()`
- Line 120: Queries Cortex MCP via `cortex.query()`
- Lines 152-168: Creates round packets for tracking

### Test Execution Status
```bash
# Current status
$ pytest tests/integration/test_agent_persona_with_cortex.py -v
# Result: 4 deselected (pytest.ini excludes integration tests)

# After pytest.ini update
$ pytest tests/integration/test_agent_persona_with_cortex.py -v -m integration
# Expected: 4 passed
```

## Next Steps

1. Update `pytest.ini` to enable integration tests
2. Run tests and fix any failures
3. Delete old `.disabled` file
4. Mark task as DONE
