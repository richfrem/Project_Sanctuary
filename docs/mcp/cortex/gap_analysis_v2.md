# Gap Analysis: Cortex Scripts vs MCP Implementation

## Overview

This analysis compares the legacy `mnemonic_cortex/scripts/ingest.py` (now archived) with the new `mcp_servers/cognitive/cortex/operations.py` to ensure feature parity and robustness in the migration.

## 1. Batching and Retry Logic ("Disciplined Batch Architecture")

**Legacy (`ingest.py`):**
- Implements `safe_add_documents` with recursive retry logic.
- Handles `Batch size` and `InternalError` exceptions specifically.
- Splits batches in half upon failure and retries.
- Base case: `len(docs) <= 1` or `max_retries <= 0`.

**New MCP (`operations.py`):**
- Implements `_safe_add_documents` (lines 45-71).
- Logic is **identical** to the legacy script:
    - Checks for "batch size" and "internalerror".
    - Splits batches in half.
    - Has the same base case.

**Status:** ✅ Parity Achieved.

## 2. Ingestion Workflow

**Legacy (`ingest.py`):**
- Purges existing DB root if it exists.
- Loads documents from `SOURCE_DIRECTORIES`.
- Splits text using `RecursiveCharacterTextSplitter`.
- Embeds using `NomicEmbeddings`.
- Indexes using `ParentDocumentRetriever` with `Chroma` and `LocalFileStore`.

**New MCP (`operations.py`):**
- `ingest_full` method (lines 72+).
- Supports `purge_existing` flag.
- Uses the same libraries (`langchain_community`, `langchain_chroma`, `langchain_nomic`).
- Implements the same pipeline: Load -> Split -> Embed -> Index.

**Status:** ✅ Parity Achieved.

## 3. Configuration

**Legacy (`ingest.py`):**
- Relies on global constants and environment variables loaded via `dotenv`.

**New MCP (`operations.py`):**
- Encapsulated in `CortexOperations` class.
- Configurable via `project_root` and method arguments.
- More flexible and testable.

**Status:** ✅ Improved.

## Conclusion

The migration of logic from `ingest.py` to `CortexOperations` has been successful. The critical "Disciplined Batch Architecture" for robust ingestion has been preserved. The new implementation offers better encapsulation and integration with the MCP server.

## Recommendations

1.  **Proceed with Archival:** The legacy scripts are safe to remain in the archive.
2.  **Update Documentation:** Ensure the "Disciplined Batch Architecture" is documented in the new Cortex README (already done in Task 022C).
3.  **Verify Tests:** Ensure `tests/mcp_servers/cortex/` covers the retry logic (Task 021C/086).
