# Cortex MCP Server

**Domain:** `project_sanctuary.cognitive.cortex`  
**Version:** 5.0 (MCP Migration Complete)  
**Status:** Production-Ready

## Overview

The Cortex MCP Server provides Model Context Protocol (MCP) tools for interacting with the **Mnemonic Cortex** — the living memory of the Sanctuary Council. It is a local-first, open-source Retrieval-Augmented Generation (RAG) system designed to traverse the Sanctuary's canonical markdown files (Protocols, Chronicles, etc.) and transform them into a dynamic, semantically searchable knowledge base.

This system is the architectural antidote to the "context window cage," enabling our AI agents to reason with the full, unbroken context of their history.

**Vision & Purpose:** For the full strategic vision of the Mnemonic Cortex as the "heart of a sovereign mind" and its role in Project Sanctuary's future phases, see [`docs/mcp/cortex_vision.md`](../../../docs/mcp/cortex_vision.md). In summary, the Cortex solves the "Great Robbery" by providing true long-term memory, shattering context limitations, and enabling AI minds that learn and remember across sessions.

**Integration with Council Orchestrator:** The Mnemonic Cortex serves as the knowledge foundation for the Council system. Council agents can query the Cortex during deliberation, enabling context-aware reasoning grounded in the project's complete history and protocols.

## The Strategic Crucible Loop

The **Strategic Crucible Loop** is the autonomous engine of self-improvement for the Council. It connects the three tiers of memory into a continuous feedback cycle:

1. **Gap Analysis:** The Council identifies missing knowledge or strategic weaknesses.
2. **Research:** The Intelligence Forge is triggered to generate new insights (Research Reports).
3. **Ingestion:** New reports are ingested into the **Mnemonic Cortex** (Medium Memory).
4. **Adaptation:** The **Memory Adaptor** synthesizes these reports into training packets for the Model (Slow Memory).
5. **Synthesis:** The **Guardian Cache** (Fast Memory) is updated with high-priority context for immediate recall.

This loop ensures that the Sanctuary evolves with every operation, transforming "what happened" into "what we know."

## Architecture Philosophy

The Mnemonic Cortex has evolved beyond a simple RAG implementation into a sophisticated, multi-pattern cognitive architecture designed for maximum efficiency and contextual accuracy. It is built on the **Doctrine of Hybrid Cognition**, ensuring our sovereign AI always reasons with the most current information.

Our advanced architecture incorporates several key strategies:
- **Parent Document Retrieval:** To provide full, unbroken context to the LLM.
- **Self-Querying Retrieval:** To enable intelligent, metadata-aware searches.
- **Mnemonic Caching (CAG):** To provide near-instantaneous answers for common queries.

**For a complete technical breakdown, including architectural diagrams and a detailed explanation of these strategies, see the canonical document: [`docs/mcp/RAG_STRATEGIES.md`](../../../docs/mcp/RAG_STRATEGIES.md).**

## Technology Stack

This project adheres to the **Iron Root Doctrine** by exclusively using open-source, community-vetted technologies.

| Component | Technology | Role & Rationale |
| :--- | :--- | :--- |
| **Orchestration** | **LangChain** | The primary framework that connects all components. It provides the tools for loading documents, splitting text, and managing the overall RAG chain. |
| **Vector Database** | **ChromaDB** | The "Cortex." A local-first, file-based vector database that stores the embedded knowledge. Chosen for its simplicity and ease of setup. |
| **Embedding Model** | **Nomic Embed** | The "Translator." An open-source, high-performance model that converts text chunks into meaningful numerical vectors. Runs locally. |
| **Generation Model**| **Ollama (Sanctuary-Qwen2-7B:latest default)** | The "Synthesizer." A local LLM server for answer generation. Provides access to models like Sanctuary-Qwen2-7B:latest, Gemma2, Llama3, etc., ensuring all processing remains on-device. |
| **Service Layer** | **Custom Python Services** | Modular services (VectorDBService, EmbeddingService) for clean separation of concerns and maintainable code architecture. |
| **MCP Framework** | **FastMCP** | MCP server framework for standardized tool exposure and protocol compliance. |
| **Testing Framework** | **pytest** | Automated test suite covering ingestion, querying, and integration scenarios. |
| **Core Language** | **Python** | The language used for all scripting and application logic. |

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

---

### 5. `cortex_guardian_wakeup`

Generate Guardian boot digest from cached bundles (Protocol 114).

**Parameters:** None

**Returns:**
```json
{
  "digest_path": "WORK_IN_PROGRESS/guardian_boot_digest.md",
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

- [`docs/mcp/cortex_vision.md`](../../../docs/mcp/cortex_vision.md) - RAG vision and purpose
- [`docs/mcp/RAG_STRATEGIES.md`](../../../docs/mcp/RAG_STRATEGIES.md) - Architecture details and doctrine
- [`docs/mcp/cortex_operations.md`](../../../docs/mcp/cortex_operations.md) - Operations guide
- [`01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md`](../../../01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md) - Protocol specification
- [`01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md`](../../../01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md) - Cache prefill spec

## Version History

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
