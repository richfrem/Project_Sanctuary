# RAG Cortex MCP Tests

Server-specific tests for RAG Cortex, verifying vector database operations, ingestion, and retrieval.

## Structure

### 1. Unit Tests (`unit/`)
Tests data models, validators, and error handling without external dependencies.

### 2. Integration Tests (`integration/`)
**File:** `test_operations.py`
- **Primary Test Suite.**
- Validates all Cortex tools (`ingest_incremental`, `query`, `get_stats`, `cache_*`) against a **REAL** ChromaDB instance.
- Uses `BaseIntegrationTest` to ensure ChromaDB is available.
- Uses **Isolated Test Collections** (e.g., `test_child_<timestamp>`) to perfectly isolate tests from each other and from the main database.

### 3. E2E Tests (`e2e/`)
**File:** `test_pipeline.py`
- End-to-end pipeline validation (formerly `test_end_to_end_pipeline.py`).
- Verifies complex workflows involving real data structures or system-level simulations.

## Prerequisites

Integration and E2E tests require ChromaDB (port 8110 for host-mapped Cortex or 8000 internal).
Embeddings are generated LOCALLY using HuggingFace (nomic-embed-text-v1.5) and do NOT require Ollama or remote APIs.
Ollama (port 11434) is only required for Forge/Reasoning tests.

```bash
# Start required services
podman compose up -d vector_db
ollama serve
```

## Running Tests

```bash
# Run all RAG Cortex tests (Unit + Integration + E2E)
pytest tests/mcp_servers/rag_cortex/ -v
```

Tests will automatically **SKIP** if services are not available, ensuring CI stability.
