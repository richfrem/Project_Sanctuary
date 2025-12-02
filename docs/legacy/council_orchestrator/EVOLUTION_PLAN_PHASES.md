# **Sanctuary Council — Evolution Plan (Phases 2 → 3 → Protocol 113)**

**Version:** 1.0
**Status:** Authoritative Roadmap
**Location:** `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`

This document defines the remaining phases of the Sanctuary Council cognitive architecture evolution. It is the official roadmap for completing the transition from a single-round orchestrator to a fully adaptive, multi-layered cognitive system based on Nested Learning principles.

---

# ✅ **Phase Overview**

There are three remaining phases, which must be completed **in strict order**:

1. **Phase 2 – Self-Querying Retriever** *(current)*
2. **Phase 3 – Mnemonic Caching (CAG)** *(next)*
3. **Protocol 113 – Council Memory Adaptor** *(final)*

Each phase enhances a different tier of the Nested Learning architecture:

| Memory Tier    | System Component       | Phase                         |
| -------------- | ---------------------- | ----------------------------- |
| Slow Memory    | Council Memory Adaptor | Protocol 113                  |
| Medium Memory  | Mnemonic Cortex        | (Supported across all phases) |
| Fast Memory    | Mnemonic Cache (CAG)   | Phase 3                       |
| Working Memory | Council Session State  | Always active                 |

---

# -------------------------------------------------------

# ✅ **PHASE 2 — Self-Querying Retriever (IN PROGRESS)**

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

# ✅ **Location Reminder**

Save this file here:

```
mnemonic_cortex/EVOLUTION_PLAN_PHASES.md
```

It will be preserved with all future `command.json` Git operations and automatically indexed by the Mnemonic Cortex.

---

If you'd like, I can also:

✅ generate a `command.json` that commits this file
✅ create a `docs/` version
✅ include it into your RAG doctrine
✅ convert it into a Mermaid roadmap diagram

Just tell me.
