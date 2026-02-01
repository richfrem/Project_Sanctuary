# TASK: Refactor RAG Ingest CLI Architecture

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** 
- `docs/architecture/mcp/servers/rag_cortex/SETUP.md`
- `mcp_servers/gateway/clusters/sanctuary_cortex/README.md`
- `00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md`

---

## 1. Objective

Create a clean CLI model for RAG ingest operations that aligns with workflow-ingest.md, consuming code from mcp_servers/rag_cortex/ via a new cli.py pattern similar to the bundling architecture refactor.

**Note:** This is a CLI UX improvement, NOT an architectural fix. The parent/child architecture is working correctly.

## 2. Deliverables

1. New `tools/ingest/cli.py` (or `tools/retrieve/ingest/cli.py`)
2. Refactored ingest commands
3. Updated `workflow-ingest.md`

## 3. Acceptance Criteria

- CLI commands work with existing parent/child chunk architecture (child_chunks_v5, parent_documents_v5)
- Cleaner interface than current MCP-based approach
- Backward compatible with cortex_cli.py ingest command

## 4. Architecture Notes (Discovered 2026-02-01)

**Current Architecture (WORKING):**
```
Query → ChromaDB (child_chunks_v5) → Match → FileStore (parent_documents_v5) → Full Context
           ↑ 400-char chunks                     ↑ 2000-char chunks
           ↑ semantic search                     ↑ complete context
```

**Key Files:**
| Component | File | Purpose |
|-----------|------|---------|
| Operations | `mcp_servers/rag_cortex/operations.py` | Core ingest/query logic |
| Child Collection | `CHROMA_CHILD_COLLECTION` (.env) | 400-char semantic chunks in ChromaDB |
| Parent Store | `CHROMA_PARENT_STORE` (.env) | 2000-char context docs in FileStore |
| Inspector | `tests/mcp_servers/rag_cortex/inspect_chroma.py` | Health check tool |
| CLI Entry | `scripts/cortex_cli.py` | Current CLI interface |

**Evidence of Working Architecture:**
- Chronicle #333: "Indexed in child_chunks_v5 and parent_documents_v5"
- Stats output: child_chunks: 2145, parent_documents: 459
- Both splitters configured: child (400/50), parent (2000/200)

## 5. Implementation Guide (Added from Code Analysis)

**Source Module:** `mcp_servers.rag_cortex.operations.CortexOperations`
- **Instantiation:** `ops = CortexOperations(PROJECT_ROOT)`
- **Core Method:** `ingest_incremental(file_paths, metadata, skip_duplicates)`
- **Validation:** Use `mcp_servers.rag_cortex.validator.CortexValidator.validate_ingest_incremental` before execution.

**Integration Strategy:**
1.  **Import:** `from mcp_servers.rag_cortex.operations import CortexOperations`
2.  **Request Model:** Map CLI args to `mcp_servers.rag_cortex.models.IngestIncrementalRequest`.
    - `file_paths`: List[str] from glob/find
    - `metadata`: Optional dict
    - `skip_duplicates`: Boolean flag
3.  **Environment:** Ensure `CHROMA_DB_DIR` and collection names (`CHROMA_CHILD_COLLECTION`, `CHROMA_PARENT_STORE`) are loaded via `mcp_servers.lib.env_helper`.
4.  **Stats:** Call `ops.get_stats()` for post-ingest summary.
