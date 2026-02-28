# RAG Cortex Agent Plugin Integration Server Documentation

## Overview

RAG Cortex Agent Plugin Integration provides retrieval-augmented generation capabilities for Project Sanctuary. It manages the knowledge base, vector embeddings, and semantic search.

## Architectural Shift: From Local File to Network Host

The RAG Cortex transitioned from a legacy file-system-based database connection to a persistent **Network Service Model** (Protocol P114).

* **Legacy Model (Deprecated):** The system stored ChromaDB files directly on disk at a path (e.g., `mnemonic_cortex/chroma_db`). This was fragile, slow, and incompatible with distributed agent architecture.
* **Current Agent Plugin Integration Model:** ChromaDB runs as a dedicated server (`vector_db` service in Docker Compose). The RAG Cortex Agent Plugin Integration connects to it via a **network address** defined in the root `.env` file (`CHROMA_HOST`, `CHROMA_PORT`).
* **Data Persistence:** Database files are persisted via a Docker bind mount to the host directory: **`.vector_data/`**. The core application logic *never* touches this folder; it only communicates over the network.

- **Vector Database:** ChromaDB for semantic search
- **Embeddings:** OpenAI text-embedding-3-small
- **Chunking:** Intelligent document chunking with metadata preservation
- **Caching:** Hot cache for frequently accessed documents

## Architecture

```
External LLM → Cortex Agent Plugin Integration (Server)
                    ↓
            ChromaDB (Vector Store)
            OpenAI Embeddings API
```

## Documentation

- **[[cortex_evolution|Cortex Evolution]]** - Evolution of the Cortex architecture
- **[[cortex_vision|Cortex Vision]]** - Long-term vision and roadmap
- **[[cortex_operations|Cortex Operations]]** - Detailed operation specifications
- **[[cortex_migration_plan|Cortex Migration Plan]]** - Migration from legacy architecture
- **[[cortex_gap_analysis|Cortex Gap Analysis]]** - Feature gap analysis
- **[[cortex_gap_analysis_comprehensive|Cortex Gap Analysis (Comprehensive)]]** - Detailed gap analysis
- **[[|Analysis Files]]** - Additional analysis documents

## Server Implementation

- **Server Code:** [[server.py|mcp_servers/rag_cortex/server.py]]
- **Operations:** [[operations.py|mcp_servers/rag_cortex/operations.py]]
- **Models:** [[models.py|mcp_servers/rag_cortex/models.py]]
- **Container Service:** [[docker-compose.yml|docker-compose.yml]] (`vector_db` service)

## Setup & Installation

For complete setup instructions, including Podman installation, service configuration, and initial database population, see:

**📖 [[SETUP|RAG Cortex Setup Guide]]**

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

- **Test Suite:** [[|tests/mcp_servers/rag_cortex/]]
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
