# TASK: Task 105: Holistic Guardian Wakeup (Context Synthesis v2)

**Status:** in-progress
**Priority:** Critical
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** 01_PROTOCOLS/118_Agent_Session_Initialization_and_MCP_Tool_Usage_Protocol.md, 01_PROTOCOLS/121_Canonical_Knowledge_Synthesis_Loop.md, 01_PROTOCOLS/123_Autonomous_Learning_Doctrine_Implementation.md

---

## 1. Objective

Implement Holistic Guardian Wakeup v2. Transition from retrieval to 'Context Synthesis Engine'. Generate high-signal 'Daily Intelligence Briefing' with synthesized Strategy, Priorities, Health, and Recency.

## 2. Deliverables

1. Updated `mcp_servers/rag_cortex/operations.py` with hybrid retrieval logic.
2. Updated Protocol 118 reflecting the new wakeup capability.
3. ADR 051: Hybrid Context Retrieval.

## 3. Acceptance Criteria

- Briefing follows 'Guardian Wakeup Briefing v2.0' schema.
- Strategic Directives include 'Gemini Signal' & 'DCD'.
- System Health uses Traffic Light (Green/Yellow/Red).
- Protocol 118 uses single 'context_briefing = cortex_guardian_wakeup(mode="HOLISTIC")' call.
- Latency < 3s.
