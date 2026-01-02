# TASK: Implement Protocol 120 - Hybrid Orchestration Framework

**Status:** complete
**Priority:** Medium
**Lead:** Claude (AI Research)
**Dependencies:** Protocol 117 (Orchestration Patterns)
**Related Documents:** docs/architecture/mcp/analysis/microsoft_agent_analysis.md

---

## 1. Objective

Formalize the distinction between LLM-driven (agentic) and workflow-driven (deterministic) orchestration, allowing for hybrid workflows that balance creativity with reliability.

## 2. Deliverables

1. **Orchestration Mode Definitions:** Clear definitions and configuration for "Agentic" vs. "Deterministic" modes.
2. **Hybrid Workflow Engine:** Support for workflows that mix both modes (e.g., deterministic setup -> agentic creative step -> deterministic validation).
3. **Guidelines:** Documentation on when to use each mode.

## 3. Acceptance Criteria

-   [ ] Define interfaces for Deterministic vs. Agentic steps.
-   [ ] Implement a workflow engine capable of executing mixed steps.
-   [ ] Create a sample hybrid workflow (e.g., deployment pipeline with an agentic review step).

## Notes

**Status Change (2025-12-06):** backlog → backlog
Moved back from in-progress as no active work is happening. Protocol 120 does not exist yet.

**Status Change (2025-12-14):** backlog → complete
Superseded by Test Pyramid Architecture (deterministic unit tests + agentic integration/E2E), Protocol 125 (5-step recursive loop + Gardener Protocol), and MCPServerFleet (deterministic lifecycle + agentic execution). Hybrid orchestration is implemented and tested in test_protocol_056_headless.py.
