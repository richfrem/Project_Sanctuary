# Cortex Gap Analysis: Legacy Scripts vs. MCP Implementation

**Objective:** Identify discrepancies between the legacy `mnemonic_cortex` scripts (proven to work) and the new `mcp_servers/cognitive/cortex` implementation.

## 1. Core Ingestion Logic (`ingest.py` vs `cortex_ingest_full`)

| Feature | Legacy (`ingest.py`) | MCP (`CortexOperations.ingest_full`) | Gap |
| :--- | :--- | :--- | :--- |
| **Batching Strategy** | "Disciplined Batch Architecture" (50 parent docs/batch) | Delegates to `IngestionService` (which *should* have it, but implementation differs) | **CRITICAL:** MCP wrapper delegates to `IngestionService` which might be masking the batching logic or error handling. Legacy script implements batching *directly* in `main()`. |
| **Error Handling** | Recursive retry (`safe_add_documents`) on ChromaDB errors | Delegates to `IngestionService` | **CRITICAL:** The robust `safe_add_documents` retry logic is defined in `ingest.py` but NOT in `IngestionService`? (Need to verify if `IngestionService` has it). |
| **ChromaDB Client** | Direct `chromadb` client management | Via `IngestionService` | Potential configuration mismatch (v4 vs v5 collections). |
| **Output** | Detailed console logs per batch | JSON summary | MCP loses visibility into batch progress. |

**Findings from Code Review:**
- `ingest.py` defines `safe_add_documents` locally.
- `IngestionService` (used by MCP) *also* defines `_safe_add_documents`.
- **Discrepancy:** `ingest.py` uses `chunked_iterable` to process batches of 50. `IngestionService.ingest_full` *also* uses `_chunked_iterable` and `_safe_add_documents`.
- **Why did MCP fail?** The `IngestionService.ingest_full` method returns `chunks_created: 0` hardcoded!
  ```python
  # IngestionService.ingest_full
  return {
      "documents_processed": total_docs,
      "chunks_created": 0, # Difficult to count exactly...
      ...
  }
  ```
- **Conclusion:** The "0 chunks" output was a red herring. The ingestion likely *worked*, but the reporting was flawed. However, the *indexing* of Protocol 101 v3.0 failed in MCP but worked in legacy. This suggests a configuration or environment difference (e.g., `DB_PATH` or `CHROMA_ROOT` resolution).

## 2. Incremental Ingestion (`ingest_incremental.py` vs `cortex_ingest_incremental`)

| Feature | Legacy | MCP | Gap |
| :--- | :--- | :--- | :--- |
| **Logic** | Checks duplicates, adds docs | Delegates to `IngestionService` | Seemingly aligned, but MCP wrapper adds abstraction layer. |
| **Reporting** | Detailed | JSON | MCP reporting is consistent with `IngestionService`. |

## 3. Missing Capabilities (Scripts without MCP Equivalents)

| Legacy Script | Purpose | MCP Equivalent | Action |
| :--- | :--- | :--- | :--- |
| `inspect_db.py` | Debugging/Viewing DB content | `cortex_get_stats` (partial) | **Add `cortex_inspect` tool?** or rely on `cortex_query`. |
| `verify_all.py` | Verification suite | None | **Migrate to `tests/mcp_servers/cortex/`** as a test suite. |
| `cache_warmup.py` | Pre-populating cache | `cortex_cache_warmup` | **Implemented.** |
| `protocol_87_query.py` | Specific protocol query | `cortex_query` | **Covered by general query.** |
| `train_lora.py` | Fine-tuning | None | **Out of scope for MCP?** Or add `cortex_train`? |

## 4. Configuration & Environment

- **Legacy:** Uses `dotenv` to load from project root. Logic for `CHROMA_ROOT` is complex/redundant in both places.
- **MCP:** Also loads `dotenv`.
- **Risk:** If `mcp_server` runs with a different CWD or env, it might point to a different DB instance.

## 5. Recommendations

1.  **Refactor `CortexOperations`:** Stop delegating to `IngestionService`. Port the *exact* logic from `ingest.py` (including the robust `safe_add_documents` and batching) directly into `operations.py`. This removes the "middleman" service which is causing confusion and reporting errors.
2.  **Fix Reporting:** Ensure `chunks_created` is actually counted (or at least estimated) so we know if ingestion did anything.
3.  **Migrate `verify_all.py`:** This is a valuable test script. Convert it into a standard `pytest` integration test in `tests/mcp_servers/cortex/test_ingestion_integrity.py`.
4.  **Archive `IngestionService`:** Once `operations.py` is self-contained, `IngestionService` becomes redundant legacy code.
