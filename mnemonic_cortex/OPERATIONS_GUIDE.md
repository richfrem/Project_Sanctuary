# Mnemonic Cortex Operations Guide

**Version:** 1.0
**Scope:** Execution instructions for all scripts, tests, and core operations within the Mnemonic Cortex system.

## 1. Directory Structure Overview

The `mnemonic_cortex/` directory contains several key subdirectories, each with specific operational tools:

- **`app/`**: Core application logic and services.
  - `main.py`: Primary entry point for RAG queries.
- **`scripts/`**: Operational scripts for ingestion, maintenance, and training.
- **`tests/`**: Unit and integration tests.
- **`core/`**: Shared utilities and configuration.

## 2. Operational Scripts (`scripts/`)

For detailed documentation of each script, see [`scripts/README.md`](scripts/README.md).

### Quick Reference

| Operation | Script | Command |
|-----------|--------|---------|
| **Full Ingest** | `ingest.py` | `python3 mnemonic_cortex/scripts/ingest.py` |
| **Incremental Ingest** | `ingest_incremental.py` | `python3 mnemonic_cortex/scripts/ingest_incremental.py <file>` |
| **Structured Query** | `protocol_87_query.py` | `python3 mnemonic_cortex/scripts/protocol_87_query.py <json_file>` |
| **Agentic Query** | `agentic_query.py` | `python3 mnemonic_cortex/scripts/agentic_query.py "<question>"` |
| **Cache Warmup** | `cache_warmup.py` | `python3 mnemonic_cortex/scripts/cache_warmup.py` |
| **Health Check** | `inspect_db.py` | `python3 mnemonic_cortex/scripts/inspect_db.py` |
| **Chronicle Index** | `create_chronicle_index.py` | `python3 mnemonic_cortex/scripts/create_chronicle_index.py` |
| **Train LoRA** | `train_lora.py` | `python3 mnemonic_cortex/scripts/train_lora.py --data <file> --output <dir>` |

**Note:** All commands must be run from the project root: `/Users/richardfremmerlid/Projects/Project_Sanctuary`

## 3. Core Application (`app/`)

### Direct RAG Query (`main.py`)
The main application entry point can be run directly to perform RAG queries.

**Usage:**
```bash
python3 mnemonic_cortex/app/main.py "Your question here"
```

**What it does:**
- Initializes the full RAG pipeline (VectorDB, Embeddings)
- Retrieves relevant context using Parent Document Retriever
- Generates a response (if LLM is connected) or returns retrieved documents

## 4. Testing (`tests/`)

The test suite ensures system integrity. Tests are built with `pytest`.

### Running All Tests
```bash
pytest mnemonic_cortex/tests/
```

### Master Verification Harness
For a complete system check (RAG, Cache, Guardian, Training), use the master harness:
```bash
python3 mnemonic_cortex/scripts/verify_all.py
```
This script runs:
1. Database Health Check
2. RAG Query Test
3. Cache Warmup
4. Cache Operations (Get/Set)
5. Guardian Wakeup
6. Adaptation Packet Generation
7. LoRA Training Dry-Run

### Running Specific Test Categories

**1. Ingestion Service Tests**
Verifies document processing, chunking, and vector store insertion.
```bash
pytest mnemonic_cortex/tests/test_ingestion_service.py
```

**2. Cache System Tests**
Verifies CAG (Context-Aware Generation) caching mechanisms.
```bash
pytest mnemonic_cortex/tests/test_cache.py
```

**3. Vector DB Service Tests**
Verifies retrieval logic and database interactions.
```bash
pytest mnemonic_cortex/tests/test_vector_db_service.py
```

## 5. MCP Server Operations

The Mnemonic Cortex is also exposed as an MCP (Model Context Protocol) server.

**Configuration:**
Ensure `cortex` is configured in your `mcp_config.json`.

**Tools Available:**

**Core RAG:**
- `cortex_query(query, max_results=5, use_cache=False)` - Semantic search
- `cortex_ingest_incremental(file_paths, metadata=None)` - Add documents
- `cortex_ingest_full(purge_existing=True)` - Full database rebuild
- `cortex_get_stats()` - Database statistics

**Cache (CAG):**
- `cortex_cache_get(query)` - Retrieve cached answer
- `cortex_cache_set(query, answer)` - Store answer
- `cortex_cache_warmup(genesis_queries=None)` - Pre-populate cache
- `cortex_cache_stats()` - Cache hit/miss stats

**Guardian & Adaptation:**
- `cortex_guardian_wakeup()` - Generate boot digest for Guardian
- `cortex_generate_adaptation_packet(days=7)` - Create fine-tuning dataset

## 6. Troubleshooting

- **Import Errors:** Ensure `PYTHONPATH` includes the project root.
  ```bash
  export PYTHONPATH=$PYTHONPATH:.
  ```
- **Database Locks:** If ChromaDB is locked, ensure no other process (like the MCP server) is holding the lock, or restart the process.
- **Missing Dependencies:** Run `pip install -r requirements.txt`.
