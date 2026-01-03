# ADR 082: Harmonized Content Processing Architecture

**Status:** PROPOSED  
**Author:** Guardian / Antigravity Synthesis  
**Date:** 2025-12-28  
**Supersedes:** None  
**Related:** ADR 081 (Soul Dataset Structure), ADR 079 (Soul Persistence), Protocol 128 (Hardened Learning Loop)

---

## Context: The Fragmentation Problem

Project Sanctuary has evolved three distinct content processing pipelines that share overlapping concerns but use separate implementations:

| System | Location | Purpose |
|--------|----------|---------|
| **Forge Fine-Tuning** | `forge/scripts/` | Generates JSONL training data for LLM fine-tuning |
| **RAG Vector DB** | `mcp_servers/rag_cortex/operations.py` | Full/incremental ingestion into ChromaDB |
| **Soul Persistence** | `mcp_servers/lib/hf_utils.py` | Uploads snapshots to Hugging Face Commons |

### Forge Fine-Tuning Scripts (Detailed)

| Script | Purpose |
|--------|----------|
| `forge_whole_genome_dataset.py` | Parses `markdown_snapshot_full_genome_llm_distilled.txt` → JSONL |
| `validate_dataset.py` | Validates JSONL syntax, schema (`instruction`, `output`), duplicates |
| `upload_to_huggingface.py` | Uploads GGUF/LoRA/Modelfile to HF Model repos |

### Current State Analysis

**Shared Concerns (Chain of Dependency)**:

![Harmonized Content Processing](../docs/architecture_diagrams/system/harmonized_content_processing.png)

*[Source: harmonized_content_processing.mmd](../docs/architecture_diagrams/system/harmonized_content_processing.mmd)*

**Key Finding:** Forge already consumes `snapshot_utils.generate_snapshot()` output!

| Concern | snapshot_utils | RAG operations | Forge scripts | hf_utils |
|---------|----------------|----------------|---------------|----------|
| Exclusion Lists | ✅ Source | ✅ Imports | 🔄 Via snapshot | ❌ N/A |
| File Traversal | ✅ Source | ✅ Re-implements | 🔄 Via snapshot | ❌ N/A |
| Code-to-Markdown | ❌ N/A | ✅ `ingest_code_shim.py` | ❌ N/A | ❌ N/A |
| Snapshot Generation | ✅ Source | ✅ Calls | 🔄 Consumes output file | ✅ Needs |
| JSONL Formatting | ❌ N/A | ❌ N/A | ✅ `determine_instruction()` | ✅ ADR 081 |
| HF Upload | ❌ N/A | ❌ N/A | ✅ `upload_to_huggingface.py` | ✅ Source |

**Divergent Concerns (Legitimately Different)**:

| Concern | Forge | RAG | Soul |
|---------|-------|-----|------|
| **Output Format** | JSONL (`instruction`, `input`, `output`) | ChromaDB embeddings | JSONL per ADR 081 |
| **Chunking Strategy** | Document-level (whole file) | Parent/child semantic chunks | Document-level |
| **Instruction Generation** | `determine_instruction()` heuristics | N/A | N/A |
| **Destination** | Local file → HF Model repo | Vector DB | HF Dataset repo |
| **Schema Validation** | `validate_dataset.py` | Implicit | ADR 081 manifest |

### The Maintenance Burden

Every time we update exclusion patterns or improve code parsing:
1. `snapshot_utils.py` must be updated (exclusions, traversal)
2. `rag_cortex/operations.py` must import and use correctly
3. `ingest_code_shim.py` must stay aligned
4. Forge scripts duplicate much of this logic

This leads to:
- **Inconsistent behavior** between systems
- **Triple maintenance** when patterns change
- **Difficult debugging** when systems produce different results

---

## Decision Options

### Option A: Status Quo (3 Separate Implementations)

Maintain each system independently.

**Pros:**
- No refactoring required
- Each system can evolve independently

**Cons:**
- Triple maintenance burden
- Inconsistent exclusion patterns across systems
- Bug fixes must be applied in multiple places
- Difficult to ensure content parity

**Verdict:** ❌ Not recommended (technical debt accumulation)

---

### Option B: Unified Content Processing Library

Create a new shared library `mcp_servers/lib/content_processor.py` that all three systems use.

```
mcp_servers/lib/
├── content_processor.py   # [NEW] Core content processing
│   ├── ContentProcessor class
│   │   ├── traverse_and_filter()      # Unified file traversal with exclusions
│   │   ├── transform_to_markdown()    # Uses ingest_code_shim
│   │   ├── chunk_for_rag()            # Parent/child chunking
│   │   ├── chunk_for_training()       # Instruction/response pairs
│   │   └── generate_manifest_entry()  # Provenance tracking
├── exclusion_config.py    # [NEW] Single source of truth for patterns
├── ingest_code_shim.py    # [MOVE] from rag_cortex/
├── snapshot_utils.py      # [REFACTOR] to use ContentProcessor
├── hf_utils.py            # [REFACTOR] to use ContentProcessor
└── path_utils.py          # [KEEP] existing
```

**Pros:**
- Single source of truth for exclusions
- Consistent code-to-markdown transformation
- Shared chunking logic with format-specific adapters
- Bug fixes apply everywhere automatically

**Cons:**
- Significant refactoring effort
- Risk of breaking working systems
- Requires careful backward compatibility testing

**Verdict:** ✅ Recommended (long-term maintainability)

---

### Option C: Lightweight Harmonization (Extract Exclusions Only)

Minimal change: Consolidate only the exclusion patterns, keep processing separate.

```
mcp_servers/lib/
├── exclusion_config.py    # [NEW] All patterns in one place
│   ├── EXCLUDE_DIR_NAMES
│   ├── ALWAYS_EXCLUDE_FILES
│   ├── ALLOWED_EXTENSIONS
│   └── should_exclude_path()     # Unified check function
```

Update all systems to import from `exclusion_config.py`.

**Pros:**
- Low risk, minimal code changes
- Solves the most common inconsistency issue
- Can be done incrementally

**Cons:**
- Doesn't address code transformation duplication
- Doesn't address chunking duplication
- Still requires updating multiple files for traversal logic

**Verdict:** ⚡ Acceptable (quick win, but incomplete)

---

## Recommended Approach: Risk-Ordered Rollout

We adopt a **consumer-driven rollout** starting with the newest code (lowest risk) and ending with the most critical code (highest protection):

### Phase 1: Create `content_processor.py` + HF Consumer (Immediate)

**Goal:** Build the new library with HF soul persistence as the first consumer.

1. Create `mcp_servers/lib/content_processor.py` with:
   - Shared exclusion logic (from `snapshot_utils.py`)
   - Code-to-markdown transformation (from `ingest_code_shim.py`)
   - File traversal utilities
   - `.to_soul_jsonl()` adapter for ADR 081 format

2. Update `mcp_servers/lib/hf_utils.py` to use `ContentProcessor`

3. Test thoroughly with `persist_soul()` operation

**Validation:** Verify HF uploads match expected ADR 081 schema.

---

### Phase 2: Update RAG Ingestion (Short-term)

**Goal:** Migrate `rag_cortex/operations.py` to use the new library.

1. Add `.to_rag_chunks()` adapter to `ContentProcessor`
2. Refactor `ingest_full()` to use `ContentProcessor`
3. Refactor `ingest_incremental()` to use `ContentProcessor`
4. Keep `ingest_code_shim.py` as a thin wrapper (backward compatibility)

**Validation:** Compare chunk counts and content before/after migration.

---

### Phase 3: Update Forge Fine-Tuning (Long-term, Protected)

**Goal:** Migrate `forge_whole_genome_dataset.py` to use the unified library.

> ⚠️ **CAUTION:** This is the most sensitive code path. Extra validation required.

1. Add `.to_training_jsonl()` adapter with `determine_instruction()` logic
2. Refactor `forge_whole_genome_dataset.py` to call `ContentProcessor`
3. Run `validate_dataset.py` before AND after to verify parity
4. Keep original script logic available for rollback

**Validation:** Byte-for-byte comparison of JSONL output with previous version.

---

## Architecture Diagram

*(See harmonized diagram above)*

---

## Implementation Considerations

### Backward Compatibility

All existing function signatures must remain supported:
- `snapshot_utils.generate_snapshot()` → Continue working as-is
- `rag_cortex.ingest_code_shim.convert_and_save()` → Re-export from new location
- `hf_utils.upload_soul_snapshot()` → No interface change

### Testing Strategy

| Phase | Test Type | Scope |
|-------|-----------|-------|
| Phase 1 | Unit tests for `should_exclude_path()` | All exclusion patterns |
| Phase 2 | Integration tests for code-to-markdown | Python, JS, TS file parsing |
| Phase 3 | E2E tests for each consumer | RAG ingestion, Forge output, HF upload |

### Fine-Tuning Code Safety

> **CAUTION (Per User Request):** Fine-tuning JSONL generation is the highest-risk area.

The Forge scripts that generate training data must:
1. Never be modified without explicit testing
2. Use the shared library **in addition to** existing validation
3. Maintain a separate manifest for training data provenance

---

## Consequences

### Positive

- **Single Source of Truth**: Exclusion patterns maintained in one file
- **Consistent Behavior**: All systems use identical filtering logic
- **Reduced Maintenance**: Bug fixes apply once, affect all consumers
- **Better Testing**: Consolidated logic enables comprehensive unit tests
- **Cleaner Architecture**: Clear separation of concerns

### Negative

- **Migration Effort**: Phase 2-3 requires significant refactoring
- **Risk During Transition**: Potential for breaking changes
- **Import Complexity**: More cross-module dependencies

### Mitigations

- Phased approach reduces risk
- Comprehensive testing before each phase
- Backward-compatible wrappers during transition

---

## Decision

**Selected Option:** Phased Harmonization (C → B)

**Rationale:** Start with low-risk extraction (Phase 1), prove value, then proceed to deeper consolidation. This balances immediate wins against long-term architectural goals.

---

## Action Items

| Task | Phase | Priority | Status |
|------|-------|----------|--------|
| Create `content_processor.py` | 1 | P1 | ⏳ Pending |
| Add `.to_soul_jsonl()` adapter | 1 | P1 | ⏳ Pending |
| Refactor `hf_utils.py` to use ContentProcessor | 1 | P1 | ⏳ Pending |
| Test `persist_soul()` with new processor | 1 | P1 | ⏳ Pending |
| Add `.to_rag_chunks()` adapter | 2 | P2 | ⏳ Pending |
| Refactor `ingest_full()` | 2 | P2 | ⏳ Pending |
| Refactor `ingest_incremental()` | 2 | P2 | ⏳ Pending |
| Add `.to_training_jsonl()` adapter | 3 | P3 | ⏳ Pending |
| Refactor `forge_whole_genome_dataset.py` | 3 | P3 | ⏳ Pending |
| Comprehensive test suite | All | P1 | ⏳ Pending |

---

## Related Documents

- [ADR 079: Soul Persistence via Hugging Face](./079_soul_persistence_hugging_face.md)
- [ADR 081: Soul Dataset Structure](./081_soul_dataset_structure.md)
- [Protocol 128: Hardened Learning Loop](../01_PROTOCOLS/128_Hardened_Learning_Loop.md)
- [ingest_code_shim.py](../mcp_servers/rag_cortex/ingest_code_shim.py)
- [snapshot_utils.py](../mcp_servers/lib/snapshot_utils.py)

---

*Proposed: 2025-12-28 — Awaiting Strategic Review*
