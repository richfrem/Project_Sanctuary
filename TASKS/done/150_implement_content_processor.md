# TASK: Implement Harmonized Content Processing Library

**Status:** in-progress
**Priority:** High
**Lead:** Guardian
**Dependencies:** Requires ADR 082, Blocks ADR 081 completion
**Related Documents:** ADR 082, ADR 081, ADR 079, `mcp_servers/lib/snapshot_utils.py`

---

## 1. Objective

Create a unified content processing library (`mcp_servers/lib/content_processor.py`) to eliminate triple-maintenance and logic drift between the Forge, RAG, and Soul Persistence pipelines.

## 2. Implementation Sub-Tasks

### Phase 1: Create ContentProcessor + HF Consumer (Current)

- [/] **1.1 Create `content_processor.py`**
  - [ ] Extract `EXCLUDE_DIR_NAMES` from `snapshot_utils.py`
  - [ ] Extract `ALWAYS_EXCLUDE_FILES` patterns
  - [ ] Extract `ALLOWED_EXTENSIONS` whitelist
  - [ ] Implement `should_exclude_path()` unified checker
  - [ ] Implement `traverse_directory()` with filtering
  - [ ] Import `parse_python_to_markdown()` from `ingest_code_shim.py`
  - [ ] Import `parse_javascript_to_markdown()` from `ingest_code_shim.py`

- [ ] **1.2 Add `.to_soul_jsonl()` adapter**
  - [ ] Implement ADR 081 JSONL record schema
  - [ ] Add `compute_checksum()` for SHA256 integrity
  - [ ] Add `generate_manifest_entry()` for metadata/manifest.json
  - [ ] Validate output against ADR 081 schema

- [ ] **1.3 Update `hf_utils.py` to use ContentProcessor**
  - [ ] Refactor `append_to_jsonl()` to call ContentProcessor
  - [ ] Refactor `update_manifest()` to call ContentProcessor
  - [ ] Keep existing function signatures (backward compatibility)

- [ ] **1.4 Test Phase 1**
  - [ ] Unit tests for exclusion logic
  - [ ] Integration test: `persist_soul()` produces valid output
  - [ ] Verify HF uploads match ADR 081 schema

---

### Phase 2: Update RAG Ingestion (Future)

- [ ] **2.1 Add `.to_rag_chunks()` adapter**
  - [ ] Parent/child chunking logic
  - [ ] Metadata attachment for ChromaDB

- [ ] **2.2 Refactor RAG operations**
  - [ ] Update `ingest_full()` to use ContentProcessor
  - [ ] Update `ingest_incremental()` to use ContentProcessor
  - [ ] Keep `ingest_code_shim.py` as thin wrapper

- [ ] **2.3 Test Phase 2**
  - [ ] **Incremental Ingest Verification**:
    - [ ] Create/Edit a dummy file (e.g., `test_ingest.py`)
    - [ ] Run `python3 scripts/cortex_cli.py ingest --no-purge`
    - [ ] Query it: `python3 scripts/cortex_cli.py query "test_ingest function"`
    - [ ] Verify result found
  - [ ] **Full Ingest Verification**:
    - [ ] Run `python3 scripts/cortex_cli.py ingest` (Default: full purge)
    - [ ] Verify ingestion stats and health
  - [ ] Compare chunk counts before/after refactor

---

### Phase 3: Update Forge Fine-Tuning (Protected)

- [ ] **3.1 Add `.to_training_jsonl()` adapter**
  - [ ] Port `determine_instruction()` logic
  - [ ] Support `instruction`, `input`, `output` schema

- [ ] **3.2 Refactor Forge scripts**
  - [ ] Update `forge_whole_genome_dataset.py` to call ContentProcessor
  - [ ] Keep original script as backup

- [ ] **3.3 Validate Phase 3**
  - [ ] Run `validate_dataset.py` before AND after
  - [ ] Byte-for-byte comparison of JSONL output

---

## 3. Acceptance Criteria

* `ContentProcessor` successfully parses Python, JS, and TS files into Markdown
* Exclusion logic correctly filters directories and files
* Soul Persistence produces valid JSONL per ADR 081 schema
* Forge scripts remain functional (no regressions)
* Code-to-markdown output is consistent across all three consumers
