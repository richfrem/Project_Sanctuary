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
