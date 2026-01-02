# Implementation Plan: Migrate and Archive Legacy Mnemonic Cortex (Task #083)

## Goal
Migrate the legacy `mnemonic_cortex` architecture to the new MCP-first architecture. This ensures the Cortex MCP (`mcp_servers/cognitive/cortex`) is the single source of truth for RAG operations, possessing the robust batching and error handling logic of the legacy scripts.

## User Review Required
> [!IMPORTANT]
> **Archival:** The `mnemonic_cortex/` directory (excluding the actual database) will be moved to `ARCHIVE/`. All future RAG operations must use the Cortex MCP.

## Proposed Changes

### 1. Refactor Cortex MCP Operations
#### [MODIFY] [mcp_servers/cognitive/cortex/operations.py](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/mcp_servers/cognitive/cortex/operations.py)
- **Port Batching Logic:** Implement the `chunked_iterable` and `safe_add_documents` (recursive retry) logic directly from `ingest.py`.
- **Remove Middleware:** Remove dependency on `IngestionService`. The `CortexOperations` class will handle ingestion logic directly to ensure visibility and error handling parity with the legacy script.
- **Fix Reporting:** Ensure `chunks_created` is accurately calculated or estimated (unlike the hardcoded 0 in the current service).

### 2. Migrate Documentation
#### [NEW] [docs/architecture/mcp/cortex/](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/docs/architecture/mcp/cortex/)
- Move `mnemonic_cortex/README.md` -> `docs/architecture/mcp/cortex/README.md`
- Move `mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md` -> `docs/architecture/mcp/cortex/RAG_STRATEGIES.md`
- Update links and references.

### 3. Migrate Tests
#### [NEW] [tests/mcp_servers/cortex/](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/tests/mcp_servers/cortex/)
- Move `mnemonic_cortex/scripts/verify_all.py` logic to `tests/mcp_servers/cortex/test_ingestion_integrity.py`.
- Ensure `test_cortex_ops.py` covers the new batching logic.

### 4. Archive Legacy Code
#### [MOVE] `mnemonic_cortex/` -> `ARCHIVE/mnemonic_cortex/`
- **Exception:** The `chroma_db` (or configured DB path) must remain or be moved to a standard data location if it's currently inside `mnemonic_cortex`.
- **Decision:** We will move the *code* (scripts, app, core) to archive. We will verify where the DB lives. If it's `mnemonic_cortex/chroma_db`, we should move it to `data/chroma_db` or similar to separate code from state.

## Verification Plan

### Automated Tests
1. **Unit Tests:** Run `pytest tests/mcp_servers/cortex/` to verify the refactored operations.
2. **Ingestion Test:** Run `cortex_ingest_full` via MCP tool and verify it completes without error and indexes all documents.

### Manual Verification
1. **Protocol 101 Check:** Query `mcp5_cortex_query("Protocol 101 v3.0 content")` to confirm the specific issue is resolved.
2. **File Check:** Verify `mnemonic_cortex` is gone (except potentially the DB) and `ARCHIVE/mnemonic_cortex` exists.
