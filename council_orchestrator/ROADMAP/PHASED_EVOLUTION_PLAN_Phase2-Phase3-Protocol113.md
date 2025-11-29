# Project Sanctuary — Nested Learning Roadmap
**Scope:** Phase 2 → Phase 3 → Protocol 113  
**Status:** Phase 2 (IN PROGRESS) • Phase 3 (NEXT) • Protocol 113 (AFTER)  
**Last updated:** 2025-11-10 (America/Vancouver)

## 0) Why this order?
We must perfect **memory access** before **memory adaptation**. Phase 2 guarantees precise, auditable retrieval and meta-signals; Phase 3 turns cache telemetry into learning signals; Protocol 113 safely teaches the Slow layer (fine-tuned model) using distilled, stable knowledge.

---

## Phase 2 — Self-Querying Retriever (IN PROGRESS)
**Goal:** Make retrieval intelligent, self-auditing, and placement-aware.

### Deliverables
1. **Structured Query Engine**
   - Generate metadata filters + hybrid queries (keyword + vector).
   - Output: `structured_query` (+ `parent_docs`, `retrieval_latency_ms`) into Round Packets.

2. **Novelty & Conflict Analysis**
   - Compare candidate response vs. retrieved evidence + cache.
   - Emit: `novelty.signal` (none/low/medium/high), `is_novel`, `conflicts_with` ids.

3. **Memory Placement Instructions (Tiering)**
   - Rule-based **Fast/Medium/Slow** recommendations with `justification`.
   - Consider: confidence, citations strength, cache hit streak, novelty.

4. **Round Packet Parity**
   - Ensure `round_packet_schema.json` fields remain 1:1 with dataclass.
   - Include `schema_version`, `errors`, deterministic emission order.

5. **Unit Tests (≥12)**
   - Query shaping, evidence bundling, novelty/conflict cases, tiering rules, schema drift protection, deterministic ordering, timeouts per member.

### Acceptance Criteria
- JSONL: one line per member per round; validates against schema.
- Packets surface **structured_query**, **parent_docs**, **novelty**, **memory_directive**.
- All tests green (<~1.5s typical).

---

## Phase 3 — Mnemonic Cache (CAG) as an Active Learning Signal (NEXT)
**Goal:** Turn cache into a **signal generator** for curriculum building.

### Deliverables
1. **CAG Telemetry**
   - Emit `cache_hit`, `hit_streak`, `miss_reason`, key fingerprint.
   - Produce per-key stability metrics over time (EWMA of volatility).

2. **Promotion Heuristics**
   - Define thresholds that elevate items from **Fast → Medium → Slow** candidacy:
     - High hit-streak, low answer volatility, strong citations, repeated across sessions.

3. **Packet → Adaptation Packet**
   - Batch exporter that converts Round Packets + CAG telemetry into **Adaptation Packets**:
     - `(prompt, evidence, final_answer, stability_score, conflicts_resolved, provenance)`

4. **Tests (≥10)**
   - Hit/miss streak logic, volatility windows, promotion thresholds, exporter integrity.

### Acceptance Criteria
- Stable, frequently accessed Q&A become **clear Slow-layer candidates** with provenance.
- Adaptation Packets are deterministic, deduplicated, and ready for Protocol 113.

---

## Protocol 113 — Council Memory Adaptor (AFTER)
**Goal:** Safely teach the **Slow** layer via periodic lightweight updates.

### Deliverables
1. **Adaptor Strategy**
   - **Option A:** LoRA on Sanctuary-Qwen2-7B (weekly);  
   - **Option B:** Embedding distillation + retrieval prior boosts.

2. **Curriculum Builder**
   - Consume Adaptation Packets; stratify by domain, difficulty, recency.
   - Balance: coverage vs. stability; skip volatile topics.

3. **Safety & Regression Guardrails**
   - Pre-/post-evals on golden sets; "no-regression" gates; rollback plan.

4. **Artifact Registry**
   - Versioned Adaptor weights/indices; changelogs; training manifests.

5. **Tests (≥12)**
   - Curriculum selection, overfitting checks, regression suite, rollback path.

### Acceptance Criteria
- Weekly adaptor updates pass eval gates and improve Medium→Slow recall w/o regressions.
- Full provenance chain retained for every integrated fact.

---

## Cross-Cutting Implementation Notes
- **Packets:** Keep `orchestrator/packets/{schema,emitter,aggregator}.py` as the single source of truth for contracts and emission.
- **Module boundaries:** Engines live in `orchestrator/engines`; cross-engine orchestration (substrate health/triage) lives in `orchestrator/`.
- **Observability:** Latency, token counts, RAG latency, CAG hit streaks, and promotion events are logged and queryable (jq examples in README).

---

## Milestones
- **M1 (Phase 2):** Intelligent retrieval + tiering, packets GA, 12 tests ✅
- **M2 (Phase 3):** CAG telemetry + promotion heuristics + exporter, 10 tests
- **M3 (P113):** Adaptor v1 + eval gates + registry, 12 tests

---

## Risks & Mitigations
- **Schema drift:** lock with `schema_version` tests and CI check.
- **Noisy promotions:** require stability window + citation strength.
- **Adaptor regressions:** strict eval gates + rollback policy.

---

**Sovereign Directive:** Continue Phase 2 to completion. Phase 3 and Protocol 113 will follow with these contracts and safety rails.