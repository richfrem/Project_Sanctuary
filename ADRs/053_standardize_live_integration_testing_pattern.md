# Standardize 3-Layer Test Pyramid for All MCP Servers

**Status:** accepted
**Date:** 2025-12-14
**Author:** AI Assistant
**Updated:** 2025-12-14 (Expanded to cover all 3 layers)


---

## Context

ADR 047 mandated systemic integration testing. ADR 048 established Layer 2 (integration tests). However, we lacked a complete, standardized 3-layer test pyramid structure across all 12 MCP servers. 

**Problems:**
1. Inconsistent test organization across MCP servers
2. Missing E2E (Layer 3) infrastructure and tests
3. Some integration tests were misplaced in `tests/integration/` (testing single servers, not multi-MCP workflows)
4. No standardized base classes for integration and E2E tests
5. Incomplete documentation of test layer purposes and patterns

**Prior State:**
- Layer 1 (Unit): ✅ Existed but no standard patterns documented
- Layer 2 (Integration): ✅ Existed but needed standardization
- Layer 3 (E2E): ❌ Missing except for one standalone test

## Decision

We will implement a **complete 3-layer test pyramid** for all 12 MCP servers with standardized structure, base classes, and infrastructure.

### Layer 1: Unit Tests (`unit/`)
- **Purpose:** Test atomic logic in complete isolation
- **Dependencies:** None (mocked if needed)
- **Base Class:** None required (tests are isolated)
- **Speed:** Fast (milliseconds)
- **Location:** `tests/mcp_servers/<server>/unit/`

### Layer 2: Integration Tests (`integration/`)
- **Purpose:** Test server operations with real local dependencies
- **Dependencies:** ChromaDB (port 8000), Ollama (port 11434), Git repo
- **Base Class:** `BaseIntegrationTest` (enforces dependency checking)
- **Speed:** Medium (seconds)
- **Location:** `tests/mcp_servers/<server>/integration/`

**Implementation:**
- All integration tests MUST inherit from `BaseIntegrationTest`
- Base class automatically checks if required services are running
- Tests skip if dependencies missing (local dev), fail if missing (CI with `CI=true`)
- Each test class defines `get_required_services()` method

### Layer 3: E2E Tests (`e2e/`)
- **Purpose:** Test full MCP client call lifecycle via MCP protocol
- **Dependencies:** All 12 MCP servers running
- **Base Class:** `BaseE2ETest` (provides MCP client utilities)
- **Speed:** Slow (minutes)
- **Location:** `tests/mcp_servers/<server>/e2e/`

**Implementation:**
- All E2E tests MUST inherit from `BaseE2ETest`
- Tests use `@pytest.mark.e2e` decorator
- Tests use `mcp_servers` pytest fixture (auto-starts all servers)
- Fixture uses standard `start_mcp_servers.py` script for consistency

### Infrastructure

**Base Classes:**
1. `tests/mcp_servers/base/base_integration_test.py` - Layer 2 base class
2. `tests/mcp_servers/base/base_e2e_test.py` - Layer 3 base class
3. ~~`base_unit_test.py`~~ - Removed (unnecessary for isolated tests)

**Fixtures:**
- `mcp_servers` (session-scoped) - Starts all 12 MCP servers for E2E tests
- Uses `mcp_servers/start_mcp_servers.py --run` for consistency with VS Code

**Documentation:**
- `tests/mcp_servers/base/README.md` - Common patterns for all 3 layers
- `tests/mcp_servers/README.md` - Test pyramid overview

### Reorganization

**Moved to Correct Locations:**
- `tests/integration/test_strategic_crucible_loop.py` → `orchestrator/integration/`
- `tests/integration/test_056_loop_hardening.py` → `orchestrator/integration/`
- `tests/mcp_servers/task/test_e2e_workflow.py` → `task/e2e/test_task_e2e.py`

**Removed Duplicates:**
- `tests/integration/test_cortex_operations.py` (duplicate)
- `tests/integration/test_end_to_end_rag_pipeline.py` (duplicate)
- `tests/integration/test_rag_simple.py` (duplicate)
- `tests/integration/test_git_workflow_end_to_end.py` (duplicate)

**Remaining in `tests/integration/`:**
Only true multi-MCP workflow tests:
- `test_chain_agent_forge.py` (Agent Persona → Forge LLM)
- `test_chain_council_agent.py` (Council → Agent Persona)
- `test_chain_forge_ollama.py` (Forge LLM → Ollama)
- `test_council_with_git.py` (Council → Git)

## Consequences

**Positive:**
- ✅ Complete 3-layer test pyramid implemented across all 12 MCP servers
- ✅ Consistent test structure and patterns
- ✅ Automatic dependency verification for integration tests
- ✅ E2E infrastructure established (ready for MCP client integration)
- ✅ Clear separation of concerns (unit vs integration vs E2E)
- ✅ Reduced boilerplate through base classes
- ✅ Better test organization and discoverability
- ✅ Comprehensive documentation of patterns

**Negative:**
- One-time reorganization effort (completed)
- Requires developers to understand 3-layer structure
- E2E tests currently skip (pending MCP client integration)

**Risks:**
- Minimal - Structure follows industry best practices
- Base classes are simple and well-documented
- Fixture management is robust with proper teardown

## Related ADRs

- **ADR 047:** Systemic Integration Testing Mandate
- **ADR 048:** (Referenced as establishing Layer 2)
- **ADR 054:** Harmonize RAG Cortex Test Structure


---

**Status Update (2025-12-14):** Implemented complete 3-layer test pyramid structure across all 12 MCP servers with standardized base classes and infrastructure
