# RAG Cortex MCP Server Documentation

## Overview

RAG Cortex MCP provides retrieval-augmented generation capabilities for Project Sanctuary. It manages the knowledge base, vector embeddings, and semantic search.

## Architectural Shift: From Local File to Network Host

The RAG Cortex transitioned from a legacy file-system-based database connection to a persistent **Network Service Model** (Protocol P114).

* **Legacy Model (Deprecated):** The system stored ChromaDB files directly on disk at a path (e.g., `mnemonic_cortex/chroma_db`). This was fragile, slow, and incompatible with distributed agent architecture.
* **Current MCP Model:** ChromaDB runs as a dedicated server (`vector_db` service in Docker Compose). The RAG Cortex MCP connects to it via a **network address** defined in the root `.env` file (`CHROMA_HOST`, `CHROMA_PORT`).
* **Data Persistence:** Database files are persisted via a Docker bind mount to the host directory: **`.vector_data/`**. The core application logic *never* touches this folder; it only communicates over the network.

- **Vector Database:** ChromaDB for semantic search
- **Embeddings:** OpenAI text-embedding-3-small
- **Chunking:** Intelligent document chunking with metadata preservation
- **Caching:** Hot cache for frequently accessed documents

## Architecture

```
External LLM → Cortex MCP (Server)
                    ↓
            ChromaDB (Vector Store)
            OpenAI Embeddings API
```

## Documentation

- **[Cortex Evolution](cortex_evolution.md)** - Evolution of the Cortex architecture
- **[Cortex Vision](cortex_vision.md)** - Long-term vision and roadmap
- **[Cortex Operations](cortex_operations.md)** - Detailed operation specifications
- **[Cortex Migration Plan](cortex_migration_plan.md)** - Migration from legacy architecture
- **[Cortex Gap Analysis](cortex_gap_analysis.md)** - Feature gap analysis
- **[Cortex Gap Analysis (Comprehensive)](cortex_gap_analysis_comprehensive.md)** - Detailed gap analysis
- **[Analysis Files](analysis/)** - Additional analysis documents

## Server Implementation

- **Server Code:** [mcp_servers/rag_cortex/server.py](../../../mcp_servers/rag_cortex/server.py)
- **Operations:** [mcp_servers/rag_cortex/operations.py](../../../mcp_servers/rag_cortex/operations.py)
- **Models:** [mcp_servers/rag_cortex/models.py](../../../mcp_servers/rag_cortex/models.py)
- **Container Service:** [docker-compose.yml](../../../../../docker-compose.yml) (`vector_db` service)

## Setup & Installation

For complete setup instructions, including Podman installation, service configuration, and initial database population, see:

**📖 [RAG Cortex Setup Guide](SETUP.md)**

Quick start:
```bash
    # 1. Ensure Podman is running (one-time setup)
    podman machine start

    # 2. Start ChromaDB container (REQUIRED)
    podman compose up -d vector_db
    
    # 3. Populate database (first time only)
    python3 mcp_servers/rag_cortex/run_cortex_integration.py --run-full-ingest

# 4. Verify
curl http://localhost:8000/api/v2/heartbeat
```

## Testing

- **Test Suite:** [tests/mcp_servers/rag_cortex/](../../../tests/mcp_servers/rag_cortex/)
- **Status:** ✅ 56/61 tests passing (5 skipped - PyTorch 3.13 compat)

## Operations

### `cortex_query`
Semantic search against the knowledge base

### `cortex_ingest_full`
Full re-ingestion (purge + rebuild) of the knowledge base

### `cortex_ingest_incremental`
Add new documents without purging existing data

### `cortex_get_stats`
Database health and statistics

### `cortex_cache_get`
Retrieve cached answer for a query

### `cortex_cache_set`
Store answer in cache for future retrieval

### `cortex_cache_stats`
Cache performance metrics

### `cortex_cache_warmup`
Pre-populate cache with genesis queries

### `cortex_guardian_wakeup`
Generate Guardian boot digest (Protocol 114)

## Performance

- **Query Latency:** <1 second for cached results
- **Ingestion:** Batched processing with retry logic
- **Cache:** Hot cache for top 100 documents

## Status

✅ **Operational** - Core functionality working, some dependency issues in tests
