# Cortex MCP Server

**Domain:** `project_sanctuary.cognitive.cortex`  
**Version:** 1.0.0  
**Status:** Phase 1 - Foundation

## Overview

The Cortex MCP Server provides Model Context Protocol (MCP) tools for interacting with the Mnemonic Cortex RAG (Retrieval-Augmented Generation) system. It exposes the knowledge base for semantic search and document ingestion.

## Architecture

This server wraps existing Mnemonic Cortex scripts and services:

- **Ingestion:** `mnemonic_cortex/scripts/ingest.py` and `ingest_incremental.py`
- **Query:** `mnemonic_cortex/app/services/vector_db_service.py` (Parent Document Retriever)
- **Stats:** Direct ChromaDB collection access

## Tools

### 1. `cortex_ingest_full`

Perform full re-ingestion of the knowledge base.

**Parameters:**
- `purge_existing` (bool, default: True): Whether to purge existing database
- `source_directories` (List[str], optional): Directories to ingest

**Returns:**
```json
{
  "documents_processed": 459,
  "chunks_created": 2145,
  "ingestion_time_ms": 45230.5,
  "vectorstore_path": "/path/to/chroma_db",
  "status": "success"
}
```

**Example:**
```python
cortex_ingest_full()
cortex_ingest_full(source_directories=["01_PROTOCOLS", "00_CHRONICLE"])
```

---

### 2. `cortex_query`

Perform semantic search query against the knowledge base.

**Parameters:**
- `query` (str): Natural language query
- `max_results` (int, default: 5): Maximum results (1-100)
- `use_cache` (bool, default: False): Use cache (Phase 2)

**Returns:**
```json
{
  "results": [
    {
      "content": "Full parent document content...",
      "metadata": {
        "source_file": "01_PROTOCOLS/101_protocol.md"
      }
    }
  ],
  "query_time_ms": 234.5,
  "cache_hit": false,
  "status": "success"
}
```

**Example:**
```python
cortex_query("What is Protocol 101?")
cortex_query("Explain the Mnemonic Cortex", max_results=3)
```

---

### 3. `cortex_get_stats`

Get database statistics and health status.

**Parameters:** None

**Returns:**
```json
{
  "total_documents": 459,
  "total_chunks": 2145,
  "collections": {
    "child_chunks": {"count": 2145, "name": "child_chunks_v5"},
    "parent_documents": {"count": 459, "name": "parent_documents_v5"}
  },
  "health_status": "healthy"
}
```

**Example:**
```python
cortex_get_stats()
```

---

### 4. `cortex_ingest_incremental`

Incrementally ingest documents without rebuilding the database.

**Parameters:**
- `file_paths` (List[str]): Markdown files to ingest
- `metadata` (dict, optional): Metadata to attach
- `skip_duplicates` (bool, default: True): Skip existing files

**Returns:**
```json
{
  "documents_added": 3,
  "chunks_created": 15,
  "skipped_duplicates": 1,
  "status": "success"
}
```

**Example:**
```python
cortex_ingest_incremental(["00_CHRONICLE/2025-11-28_entry.md"])
cortex_ingest_incremental(
    file_paths=["01_PROTOCOLS/120_new.md"],
    skip_duplicates=False
)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure MCP server in `~/.gemini/antigravity/mcp_config.json`:
```json
{
  "mcpServers": {
    "cortex": {
      "command": "python",
      "args": ["-m", "mcp_servers.cognitive.cortex.server"],
      "cwd": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      }
    }
  }
}
```

3. Restart Antigravity

## Usage

From Antigravity or any MCP client:

```
# Get database stats
cortex_get_stats()

# Query the knowledge base
cortex_query("What is Protocol 101?")

# Add a new document
cortex_ingest_incremental(["path/to/new_document.md"])

# Full re-ingestion (use with caution)
cortex_ingest_full()
```

## Safety Rules

1. **Read-Only by Default:** Query operations are read-only
2. **Ingestion Confirmation:** Full ingestion purges existing data
3. **Long-Running Operations:** Ingestion may take several minutes
4. **Rate Limiting:** Max 100 queries/minute recommended
5. **Validation:** All inputs are validated before processing

## Phase 2 Features (Upcoming)

- Cache integration (`use_cache` parameter)
- Guardian Wakeup tool (Protocol 114)
- Cache warmup and invalidation
- Cache statistics

## Dependencies

- **ChromaDB:** Vector database
- **LangChain:** RAG framework
- **NomicEmbeddings:** Local embedding model
- **FastMCP:** MCP server framework

## Related Documentation

- `mnemonic_cortex/VISION.md` - RAG vision and purpose
- `mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md` - Architecture details
- `01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md` - Protocol specification
- `01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md` - Cache prefill spec

## Version History

- **1.0.0** (2025-11-28): Phase 1 - Foundation
  - 4 core tools: ingest_full, query, get_stats, ingest_incremental
  - Parent Document Retriever integration
  - Input validation and error handling
