# Cortex MCP Server

**Description:** The **Sanctuary Cortex Cluster** is the unified cognitive engine of the system. It acts as a **Composite Gateway**, aggregating four distinct internal MCP servers into a single interface for the Orchestrator/User:
1.  **RAG Cortex**: Knowledge base and semantic search.
2.  **Learning**: Protocol 128 lifecycle and memory persistence.
3.  **Evolution**: Protocol 131 self-improvement and metrics.
4.  **Forge**: LLM reasoning and model interaction.

## Tools (Aggregated)

| Tool Name | Source | Description |
|-----------|--------|-------------|
| `cortex_query` | RAG | Semantic search against the knowledge base. |
| `cortex_ingest_full/incremental` | RAG | Ingest documents into vector store. |
| `cortex_learning_debrief` | Learning | **Protocol 128**: Generate session briefing. |
| `cortex_capture_snapshot` | Learning | **Protocol 128**: Create authorized state snapshot. |
| `cortex_persist_soul` | Learning | **ADR 079**: Broadcast learnings to Soul Genome. |
| `cortex_guardian_wakeup` | Learning | **Protocol 114**: Bootloader digest. |
| `cortex_measure_fitness` | Evolution | **Protocol 131**: Map-Elites metric calculation. |
| `query_sanctuary_model` | Forge | Query the fine-tuned Sanctuary model. |

## Resources

| Resource URI | Description | Mime Type |
|--------------|-------------|-----------|
| `cortex://stats` | Knowledge base statistics | `application/json` |
| `cortex://document/{doc_id}` | Full content of a document | `text/markdown` |

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required for Embeddings
OPENAI_API_KEY=sk-... # If using OpenAI embeddings
# Optional
CORTEX_CHROMA_DB_PATH=mcp_servers/cognitive/cortex/data/chroma_db
CORTEX_CACHE_DIR=mcp_servers/cognitive/cortex/data/cache
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"cortex": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/cognitive/cortex",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/cognitive/cortex/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `cortex_query` and `cortex_ingest_full` appear in the tool list.
3.  **Call Tool:** Execute `cortex_get_stats` and verify it returns valid JSON statistics.

## Architecture

### Overview
The Mnemonic Cortex has evolved beyond a simple RAG implementation into a sophisticated, multi-pattern cognitive architecture designed for maximum efficiency and contextual accuracy. It is built on the **Doctrine of Hybrid Cognition**, ensuring our sovereign AI always reasons with the most current information.

**Key Strategies:**
- **Parent Document Retrieval:** To provide full, unbroken context to the LLM.
- **Self-Querying Retrieval:** To enable intelligent, metadata-aware searches.
- **Mnemonic Caching (CAG):** To provide near-instantaneous answers for common queries.
- **Polyglot Code Ingestion:** Automatically converts Python and JavaScript/TypeScript files into optimize markdown for semantic indexing, using AST/regex to structurally document code without LLM overhead.

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
    file_paths=["01_PROTOCOLS/120_new.md", "mcp_servers/rag_cortex/server.py"],
    skip_duplicates=False
)
```

### Polyglot Support
The ingestion system automatically detects and converts code files:
- **Python**: Uses AST to extract classes, functions, and docstrings.
- **JS/TS**: Uses regex to extract functions and classes.
- **Output**: Generates a `.py.md` or `.js.md` companion file which is then ingested.
- **Exclusions**: Automatically skips noisy directories (`node_modules`, `dist`, `__pycache__`).
```

---

### 5. `cortex_guardian_wakeup`

Generate Guardian boot digest from cached bundles (Protocol 114).

**Parameters:** None

**Returns:**
```json
{
  "digest_path": ".agent/learning/guardian_boot_digest.md",
  "cache_stats": {
    "chronicles": 5,
    "protocols": 10,
    "roadmap": 1
  },
  "status": "success"
}
```

**Example:**
```python
cortex_guardian_wakeup()
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
- Cache warmup and invalidation
- Cache statistics

## Dependencies

- **ChromaDB:** Vector database
- **LangChain:** RAG framework
- **NomicEmbeddings:** Local embedding model
- **FastMCP:** MCP server framework

## Related Documentation

- [`docs/architecture/mcp/cortex_vision.md`](../../../../docs/architecture/mcp/servers/rag_cortex/cortex_vision.md) - RAG vision and purpose
- [`docs/architecture/mcp/RAG_STRATEGIES.md`](../../../../ARCHIVE/mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md) - Architecture details and doctrine
- [`docs/architecture/mcp/cortex_operations.md`](../../../../docs/architecture/mcp/servers/rag_cortex/cortex_operations.md) - Operations guide
- [`01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md`](../../../../01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md) - Protocol specification
- [`01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md`](../../../../01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md) - Cache prefill spec

## Version History

### v5.1 (2025-12-14): Polyglot Code Ingestion
- **Code Shim:** Introduced `ingest_code_shim.py` for AST-based code-to-markdown conversion
- **Multi-Language Support:** Added native support for .py, .js, .ts, .jsx, .tsx ingestion
- **Smart Exclusion:** Implemented noise filtering for production directories

### v5.0 (2025-11-30): MCP Migration Complete
- **Migration to MCP Architecture:** Refactored from legacy script-based system to MCP server
- **Enhanced README:** Merged legacy documentation with MCP-specific content
- **Comprehensive Documentation:** Added architecture philosophy, technology stack, and Strategic Crucible Loop context
- **Production-Ready Status:** Full test coverage and operational stability

### v2.1.0: Parent Document Retriever
- **Phase 1 Complete:** Implemented dual storage architecture eliminating Context Fragmentation vulnerability
- **Full Context Retrieval:** Parent documents stored in ChromaDB collection, semantic chunks in vectorstore
- **Cognitive Latency Resolution:** AI reasoning grounded in complete, unbroken context
- **Architecture Hardening:** Updated ingestion pipeline and query services to leverage ParentDocumentRetriever

### v1.5.0: Documentation Hardening
- **Architectural Clarity:** Added detailed section breaking down two-stage ingestion process
- **Structural Splitting vs. Semantic Encoding:** Clarified roles of MarkdownHeaderTextSplitter and NomicEmbeddings

### v1.4.0: Live Ingestion Architecture
- **Major Architectural Update:** Ingestion pipeline now directly traverses canonical directories
- **Improved Traceability:** Every piece of knowledge traced to precise source file via GitHub URLs
- **Increased Resilience:** Removed intermediate snapshot step for faster, more resilient ingestion

### v1.0.0 (2025-11-28): MCP Foundation
- **4 Core Tools:** ingest_full, query, get_stats, ingest_incremental
- **Parent Document Retriever Integration:** Full context retrieval from day one
- **Input Validation:** Comprehensive error handling and validation layer
