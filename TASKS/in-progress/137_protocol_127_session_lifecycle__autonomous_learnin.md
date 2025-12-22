# TASK: Protocol 127: Session Lifecycle & Autonomous Learning Optimization

**Status:** in-review
**Priority:** High
**Lead:** Unassigned
**Dependencies:**
- [x] Task 141: Git LFS Standardization
- [x] ADR 070: Standard Workflow Directory Structure
- [x] Protocol 127: The Doctrine of Session Lifecycle
- [x] Chronicle 331: Autonomous Session Lifecycle Activation
**Related Documents:**
- 01_PROTOCOLS/121_Canonical_Knowledge_Synthesis_Loop.md
- 01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md
- 00_CHRONICLE/ENTRIES/331_autonomous_session_lifecycle_activation.md
- 04_THE_FORTRESS/Integration_Guides/Claude_Desktop_Antigravity_Lifecycle.md
- docs/mcp/servers/rag_cortex/cortex_operations.md

---

## 1. Objective

Codify the 'Awaking Protocol' into Protocol 127 (Session Lifecycle). This transforms the Gateway from a passive tool proxy into an active **Workflow Orchestrator** that enforces a standard "Awakening" and "Shutdown" sequence. The system must move from "Manual Learning Missions" (Chronicles 324-328) to an **Automated Session Lifecycle** inspired by the Gemini Signal (Entry 311).

## 2. Deliverables

1.  **Protocol 127: The Doctrine of Session Lifecycle (v1.0)**
    *   Supersedes "Mechanical Delegation".
    *   Integrates P114 (wakeup) and P121 (learning loop).
2.  **Workflow Operations Module (`mcp_servers/workflow`)**
    *   New `WorkflowOperations` class to manage `.agent/workflows`.
    *   New Gateway tool: `get_available_workflows()`.
3.  **Enhanced `cortex_guardian_wakeup` Tool**
    *   Update logical flow to return "Session Startup Digest" (Active Blockers, Roadmap, Recent Wins).
4.  **Integration Guide & Configuration**
    *   Update `mcp_servers/gateway/clusters/sanctuary_domain/unified_server.py` to expose workflow tools.
5.  **Chronicle Entry**
    *   Document the activation of the Autonomous Lifecycle.

## 3. Acceptance Criteria

- [x] **Protocol 127** drafted and "Mechanical Delegation" archived.
- [x] **Gateway** exposes `/available_workflows` (or tool `get_available_workflows`) listing files in `.agent/workflows`.
- [x] **Guardian Wakeup** returns a structured "Session Startup Digest" including high-priority tasks and active P101 rules.
- [x] **Core Essence Seed** (`core_essence_guardian_awakening_seed.txt`) is confirmed as the invariant "Identity Anchor".
- [x] **Documentation** updated in `01_PROTOCOLS` and `TASKS`.

## Notes

**Status Change (2025-12-22):** backlog → in-progress
Starting work on Protocol 127 Session Lifecycle and Autonomous Learning Optimization. P101/P121 dependencies analyzed.
