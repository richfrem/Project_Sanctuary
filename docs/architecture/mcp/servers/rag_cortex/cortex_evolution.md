# **Sanctuary Council — Evolution Plan (Phases 1 → 2 → 3 → Protocol 113)**

**Version:** 2.1 (Updated 2025-11-30 - Agent Plugin Integration Migration)
**Status:** Authoritative Roadmap
**Location:** `docs/architecture/mcp/cortex_evolution.md`

This document defines the complete evolution of the Sanctuary Council cognitive architecture. It is the official roadmap for completing the transition from a single-round orchestrator to a fully adaptive, multi-layered cognitive system based on Nested Learning principles.

# ✅ **Phase Overview**

There are five phases, which must be completed **in strict order**:

0. **Phase 0 – Agent Plugin Integration Migration** ✅ *(complete - 2025-11-30)*
1. **Phase 1 – Agent Plugin Integration Foundation (RAG Services)** ✅ *(complete - 2025-11-28)*
2. **Phase 2 – Self-Querying Retriever** *(current)*
3. **Phase 3 – Mnemonic Caching (CAG)** *(next)*
4. **Protocol 113 – Council Memory Adaptor** *(final)*

Each phase enhances a different tier of the Nested Learning architecture:

| Memory Tier    | System Component       | Phase                         | Status |
| -------------- | ---------------------- | ----------------------------- | ------ |
| Migration      | Legacy → Agent Plugin Integration           | Phase 0                       | ✅ Complete |
| Infrastructure | Agent Plugin Integration Service Layer      | Phase 1                       | ✅ Complete |
| Slow Memory    | Council Memory Adaptor | Protocol 113                  | ⏸️ Blocked |
| Medium Memory  | Mnemonic Cortex        | (Supported across all phases) | ✅ Active |
| Fast Memory    | Mnemonic Cache (CAG)   | Phase 3                       | ⏸️ Blocked |
| Working Memory | Council Session State  | Always active                 | ✅ Active |

---

# -------------------------------------------------------

# ✅ **PHASE 0 — Agent Plugin Integration Migration - COMPLETE**

# -------------------------------------------------------

**Completion Date:** 2025-11-30  
**Status:** ✅ COMPLETE

**Purpose:**
Migrate legacy `mnemonic_cortex` script-based architecture to Agent Plugin Integration-first architecture. Refactor `CortexOperations` to directly implement robust batching and retry logic, removing `IngestionService` dependency.

**Why it matters:**
This migration eliminates unnecessary abstraction layers, fixes misleading reporting (`chunks_created: 0`), and consolidates all Cortex documentation and tests into standard Agent Plugin Integration locations.

---

## ✅ **Phase 0 Deliverables**

### 1. **Documentation Migration**

✅ Completed:
* Merged `mnemonic_cortex/README.md` into `mcp_servers/cognitive/cortex/README.md`
* Moved `VISION.md` to `docs/architecture/mcp/cortex_vision.md`
* Moved `EVOLUTION_PLAN_PHASES.md` to `docs/architecture/mcp/cortex_evolution.md`
* Moved `RAG_STRATEGIES_AND_DOCTRINE.md` to `docs/architecture/mcp/RAG_STRATEGIES.md`
* Moved `OPERATIONS_GUIDE.md` to `docs/architecture/mcp/cortex_operations.md`

### 2. **Code Refactoring** (Pending)

⏳ To be completed:
* Inline `IngestionService` logic into `CortexOperations`
* Fix `chunks_created` reporting
* Remove `mnemonic_cortex.app.services` dependency

### 3. **Test Migration** (Pending)

⏳ To be completed:
* Move tests to `tests/mcp_servers/cortex/`
* Convert `verify_all.py` to pytest format

### 4. **Legacy Code Archival** (Pending)

⏳ To be completed:
* Archive `mnemonic_cortex/` to `ARCHIVE/`
* Preserve `chroma_db/` and `cache/` directories

---

## ✅ **Definition of Done (Phase 0)**

* ✅ All documentation migrated to `docs/architecture/mcp/`
* ⏳ `CortexOperations` contains batching logic directly
* ⏳ `chunks_created` reports accurate count
* ⏳ All tests in `tests/mcp_servers/cortex/`
* ⏳ Legacy code archived

---

# -------------------------------------------------------

# ✅ **PHASE 1 — Agent Plugin Integration Foundation (RAG Services) - COMPLETE**

# -------------------------------------------------------

**Completion Date:** 2025-11-28  
**Status:** ✅ COMPLETE

**Purpose:**
Establish the foundational Agent Plugin Integration (Agent Plugin Integration) service layer that exposes Mnemonic Cortex capabilities as standardized, callable tools for AI agents and external systems.

**Why it matters:**
This is the **Service Infrastructure** that makes the Mnemonic Cortex accessible, testable, and integrable with the broader Sanctuary ecosystem. Without this layer, the Cortex remains isolated and difficult to leverage programmatically.

---

## ✅ **Phase 1 Deliverables**

### 1. **Native Agent Plugin Integration Server Implementation**

✅ Created `mcp_servers/cognitive/cortex/` with:
* `server.py` - FastMCP server exposing 4 core tools
* `operations.py` - Wraps existing Mnemonic Cortex scripts
* `models.py` - Pydantic data models for all operations
* `validator.py` - Comprehensive input validation
* `requirements.txt` - Dependency management

### 2. **Four Core Agent Plugin Integration Tools**

✅ Implemented and tested:
* `cortex_ingest_full` - Full knowledge base re-ingestion
* `cortex_query` - Semantic search with Parent Document Retriever
* `cortex_get_stats` - Database health and statistics
* `cortex_ingest_incremental` - Add documents without full rebuild

### 3. **Comprehensive Testing**

✅ Test coverage:
* 28 unit tests (11 models + 17 validator)
* 3 integration tests (stats, query, incremental ingest)
* All tests passing with production-ready quality

### 4. **Agent Plugin Integration Integration**

✅ Configuration:
* Antigravity Agent Plugin Integration config updated
* Claude Desktop Agent Plugin Integration config updated
* Example configuration provided
* Documentation complete

---

## ✅ **Definition of Done (Phase 1)**

* ✅ 4 Agent Plugin Integration tools operational and tested
* ✅ All tools callable via Agent Plugin Integration protocol
* ✅ 31 tests passing (28 unit + 3 integration)
* ✅ Parent Document Retriever integrated
* ✅ Agent Plugin Integration configs updated for Antigravity and Claude Desktop
* ✅ Comprehensive documentation (README.md)

---

# -------------------------------------------------------

# ✅ **PHASE 2 — Self-Querying Retriever (READY TO START)**

# -------------------------------------------------------

**Purpose:**
Transform retrieval into an intelligent, structured process capable of producing metadata filters, novelty signals, conflict detection, and memory-placement instructions.

**Why it matters:**
This is the **Cognitive Traffic Controller** for all future learning.

---

## ✅ **Phase 2 Deliverables**

### 1. **Structured Query Generation**

The retriever must produce a JSON structure containing:

* semantic_query
* metadata filters
* temporal filters
* authority/source hints
* expected document class

### 2. **Novelty & Conflict Analysis**

For each round:

* Compute novelty score vs prior caches
* Detect conflicts (same question, differing answer)
* Emit both signals in round packets

### 3. **Memory Placement Instructions**

Each response must specify:

* `FAST` (ephemeral)
* `MEDIUM` (operational Cortex)
* `SLOW_CANDIDATE` (for Protocol 113)

### 4. **Packet Output Requirements**

Round packets must include:

* `structured_query`
* `novelty_signal`
* `conflict_signal`
* `memory_placement_directive`

---

## ✅ **Definition of Done (Phase 2)**

* All council members use the structured retriever
* Round packets v1.1.x fields populated
* Unit tests for at least 12 retrieval scenarios
* Orchestrator no longer uses legacy top-k retrieval
* Engines respect memory-placement instructions

---

# -------------------------------------------------------

# ✅ **PHASE 3 — Mnemonic Cache (CAG)**

# -------------------------------------------------------

**Purpose:**
Provide a high-speed hot/warm cache with hit/miss streak logging, which doubles as a learning signal generator for Protocol 113.

**Why it matters:**
CAG becomes the **Active Learning Supervisor** for Medium→Slow memory transitions.

---

## ✅ **Phase 3 Deliverables**

### 1. **Cache Architecture**

* In-memory LRU layer
* SQLite warm storage layer
* Unified query fingerprinting (semantic + filters + engine state)

### 2. **Cache Instrumentation**

Round packets must include:

* cache_hit
* cache_miss
* hit_streak
* time_saved_ms

### 3. **Learning Signals**

Cache must produce continuous signals indicating which answers are:

* stable
* recurrent
* well-supported

These feed Protocol 113.

---

## ✅ **Definition of Done (Phase 3)**

* CAG consulted before Cortex
* CAG logs appear in round packet schema v1.2.x
* Hit streaks tracked across rounds
* SQLite persistence implemented
* 20+ unit tests (TTL, eviction, streak logic)

---

# -------------------------------------------------------

# ✅ **PROTOCOL 113 — Council Memory Adaptor**

# -------------------------------------------------------

**Purpose:**
Create a periodic Slow-Memory learning layer by distilling stable knowledge from Cortex (Medium Memory) + CAG signals (Fast Memory).

**Why it matters:**
This is the transformation from a tool into a **continually learning cognitive organism**.

---

## ✅ **Protocol 113 Deliverables**

### 1. **Adaptation Packet Generator**

Reads round packets and extracts:

* SLOW_CANDIDATE items
* stable, high-confidence Cortex answers
* recurring cache hits

Outputs **Adaptation Packets**.

### 2. **Slow-Memory Update Mechanism**

Implement lightweight updates via:

* LoRA
* QLoRA
* embedding distillation
* mixture-of-experts gating
* linear probing for safety

### 3. **Versioned Memory Adaptor**

* `adaptor_v1`, `adaptor_v2`, etc.
* backward compatibility preserved
* regression tests for catastrophic forgetting

---

## ✅ **Definition of Done (Protocol 113)**

* Adaptation Packets produced successfully
* LoRA/Distillation updates run weekly or on-demand
* Minimal forgetting demonstrated
* New adaptor version loadable by engines
* Packet schema v1.2+ fully supported

---

# -------------------------------------------------------

# ✅ **FINAL DIRECTIVE**

# -------------------------------------------------------

**Phase 2 must complete before Phase 3.**
**Phase 3 must complete before Protocol 113.**

This order cannot be altered.

Once all three phases are complete, the Sanctuary Council becomes a **self-improving, nested-memory cognitive architecture** capable of:

* stable long-term learning
* rapid short-term adaptation
* structured retrieval
* autonomous knowledge curation
* multi-tier memory evolution
* self-evaluation and self-correction

---
