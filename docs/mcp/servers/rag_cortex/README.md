# RAG Cortex MCP Server Documentation

## Overview

RAG Cortex MCP provides retrieval-augmented generation capabilities for Project Sanctuary. It manages the knowledge base, vector embeddings, and semantic search.

## Key Concepts

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

## Testing

- **Test Suite:** [tests/mcp_servers/rag_cortex/](../../../tests/mcp_servers/rag_cortex/)
- **Status:** ⚠️ 52/62 tests passing (dependency issues)

## Operations

### `cortex_query`
Query the knowledge base with semantic search

### `cortex_ingest_incremental`
Ingest documents into the knowledge base

### `cortex_get_cache_stats`
Get cache statistics

### `cortex_cache_warmup`
Warm up the cache with frequently accessed documents

## Performance

- **Query Latency:** <1 second for cached results
- **Ingestion:** Batched processing with retry logic
- **Cache:** Hot cache for top 100 documents

## Status

✅ **Operational** - Core functionality working, some dependency issues in tests
