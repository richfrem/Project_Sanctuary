# TASK: Post-Migration Validation: MCP Integration Testing & Naming Consistency

**Status:** Done
**Priority:** High
**Lead:** Antigravity
**Dependencies:** tasks 077, 078 complete (Agent Persona MCP and Council MCP implemented)
**Related Documents:** tests/integration/test_council_orchestrator_with_cortex.py.disabled, docs/mcp/cortex_operations.md, workflows/council_orchestration.md, mcp_servers/agent_persona/server.py, mcp_servers.rag_cortex/server.py

---

## 1. Objective

Validate and complete the council_orchestrator → Agent Persona MCP and mnemonic_cortex → Cortex MCP migration by addressing critical integration testing, naming consistency, and multi-agent deliberation verification gaps identified by architectural audit.

## 2. Deliverables

1. Refactored integration test: tests/integration/test_agent_persona_with_cortex.py
2. Updated cortex MCP server.py with consistent 'Cortex' naming
3. Documentation audit report showing all legacy references removed
4. Multi-agent deliberation test suite
5. Architecture validation report

## 3. Acceptance Criteria

- Integration test for Agent Persona MCP → Cortex MCP communication passing
- All 'RAG DB' references updated to 'Cortex' in code and docs
- Tool naming consistency verified (council_dispatch vs agent_persona_dispatch)
- Legacy council_orchestrator references removed from non-archived docs
- Multi-agent deliberation logic verified in agent_persona MCP
- Full council workflow (coordinator → strategist → auditor) tested end-to-end

## Notes

Auditor identified 3 critical post-migration validation areas:
1. Missing MCP-to-MCP integration tests (Agent Persona → Cortex)
2. Naming inconsistency ('RAG DB' vs 'Cortex', tool naming drift)
3. Multi-agent deliberation logic verification

This task ensures the architectural migration is functionally complete and properly tested.
