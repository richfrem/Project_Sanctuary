# TASK: Implement Phase 2 - Self-Querying Retriever

**Status:** BACKLOG
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Requires #017
**Related Documents:** `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`

## Context
This task is a core deliverable for the successful implementation of the Strategic Crucible Loop (Task #017), providing the intelligent retrieval necessary for the 'Automated Intelligence Forge Trigger' step.

## Objective
Transform retrieval into an intelligent, structured process by implementing a Self-Querying Retriever. This component will use an LLM to translate natural language queries into structured queries with metadata filters.

## Deliverables
1.  **Structured Query Generation:** The retriever must produce a JSON structure with `semantic_query`, `metadata_filters`, `temporal_filters`, etc.
2.  **Novelty & Conflict Analysis:** Implement logic to compare new responses against cached/retrieved data.
3.  **Memory Placement Instructions:** Generate `FAST`, `MEDIUM`, or `SLOW_CANDIDATE` directives.
4.  **Packet Output:** Ensure all new signals are correctly emitted in round packets.
