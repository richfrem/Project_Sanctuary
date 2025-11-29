# TASK: Implement Phase 3 - Mnemonic Caching (CAG)

**Status:** BACKLOG → **READY TO START** (Dependency satisfied)  
**Priority:** Medium  
**Lead:** Unassigned  
**Dependencies:** ~~Requires #050~~ ✅ **Task #050 Complete (2025-11-28)**  
**Related Documents:** `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`, `mnemonic_cortex/core/cache.py`

> [!NOTE]
> **Dependency Satisfied:** Task #050 (RAG MCP Phase 1) is complete with all 4 tools operational and tested. This task is now ready to proceed with Phase 2 (CAG cache integration).

## Context
This task is a core deliverable for the successful implementation of the Strategic Crucible Loop (Task #017), providing the telemetry and caching necessary for the 'Automated Cache Synthesis & Prefill' step.

**NOTE:** CAG cache infrastructure already exists at `mnemonic_cortex/core/cache.py` (223 lines, production-ready with hot/warm 2-tier caching). This task focuses on MCP tool integration and Protocol 114 compliance.

## Objective
Integrate existing CAG cache with RAG MCP Phase 2 tools to reduce latency and serve as a learning signal generator for Protocol 113.

## Deliverables
1.  **MCP Tools:** Implement `cortex_cache_warmup`, `cortex_cache_invalidate`, `cortex_guardian_wakeup`, `cortex_cache_stats`
2.  **Cache Instrumentation:** Emit `cache_hit`, `cache_miss`, and `hit_streak` signals in round packets.
3.  **Learning Signals:** Develop heuristics to identify stable, recurrent answers as candidates for promotion to Slow Memory.
