# TASK: Evaluate Protocol 87 Orchestrator Placement: Cortex MCP vs Council MCP

**Status:** Done
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** .agent/adr/039_mcp_server_separation_of_concerns.md, docs/mcp/RAG_STRATEGIES.md

---

## 1. Objective

Determine the optimal architectural placement for Protocol 87 structured query orchestration. Currently implemented in Cortex MCP, but Council MCP is a valid alternative. Need to evaluate trade-offs and make a decision based on separation of concerns, use cases, and future extensibility.

## 2. Deliverables

1. Architectural analysis document comparing both options
2. Decision rationale with pros/cons
3. Migration plan if Council MCP is chosen
4. Updated architecture diagrams showing chosen approach

## 3. Acceptance Criteria

- Clear decision documented with reasoning
- Architecture diagrams updated
- Implementation matches chosen architecture
- Tests verify correct MCP placement

## Notes

**Context:**
Protocol 87 orchestrator routes structured queries to specialized MCPs (Protocol, Chronicle, Task, Code, ADR).

**Option A: Cortex MCP (Current)**
- Cortex is knowledge orchestrator
- Protocol 87 is about querying knowledge
- Natural fit with RAG architecture
- Direct routing without extra hops

**Option B: Council MCP (Alternative)**
- Council is already an orchestrator (multi-agent)
- Separation: Council = orchestration, Cortex = RAG
- More general-purpose
- Could orchestrate ANY MCP workflow

**Key Question:** Is this knowledge-specific (Cortex) or general orchestration (Council)?

**Related Files:**
- mcp_servers.rag_cortex/mcp_client.py
- mcp_servers.rag_cortex/operations.py (query_structured method)
- mcp_servers/council/ (alternative location)
