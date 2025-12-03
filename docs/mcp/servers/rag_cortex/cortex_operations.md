# Mnemonic Cortex Operations Guide

**Version:** 2.0 (MCP Era)
**Scope:** Execution instructions for all operations within the Mnemonic Cortex system (now Cortex MCP).

## 1. Architecture Overview

The Mnemonic Cortex has migrated from a standalone script-based system to a fully integrated **Model Context Protocol (MCP)** server.

- **Server Location:** `mcp_servers/cognitive/cortex/`
- **Data Store:** `mcp_servers/cognitive/cortex/data/` (ChromaDB)
- **Interface:** MCP Tools (via Claude Desktop, Antigravity, or Council)

## 2. Operational Mapping (Scripts → MCP Tools)

All legacy scripts have been incorporated into the Cortex MCP. Use the corresponding MCP tools instead.

| Legacy Script | New MCP Tool | Description |
|---------------|--------------|-------------|
| `ingest.py` | `cortex_ingest_full` | Full database rebuild |
| `ingest_incremental.py` | `cortex_ingest_incremental` | Add new files |
| `protocol_87_query.py` | `cortex_query_structured` | Protocol 87 structured query |
| `inspect_db.py` | `cortex_get_stats` | Database health & stats |
| `cache_warmup.py` | `cortex_cache_warmup` | Pre-populate cache |
| `agentic_query.py` | `cortex_query` | Semantic search |
| `create_chronicle_index.py` | N/A (Handled by Chronicle MCP) | Redundant |
| `train_lora.py` | N/A (Handled by Forge MCP) | See Forge MCP |

## 3. Core Operations

### RAG Query (Semantic Search)
**Tool:** `cortex_query`
**Args:** `query` (string)
**Usage:**
> "Search the cortex for 'Protocol 101'."

### Protocol 87 Query (Structured)
**Tool:** `cortex_query_structured`
**Args:** `query_string` (Protocol 87 format)
**Usage:**
> "Execute structured query: RETRIEVE :: Protocols :: Name='Protocol 101'"

### Database Statistics
**Tool:** `cortex_get_stats`
**Usage:**
> "Get cortex database statistics."

### Ingestion
**Tool:** `cortex_ingest_incremental`
**Args:** `file_paths` (list)
**Usage:**
> "Ingest the file '01_PROTOCOLS/101_The_Doctrine.md' into cortex."

## 4. Testing & Verification

The test suite is now located in `tests/mcp_servers/cortex/`.

### Running Tests
```bash
# Run all Cortex MCP tests
pytest tests/mcp_servers/cortex/ -v
```

### Integration Testing
```bash
# Test full RAG pipeline
python3 tests/mcp_servers/cortex/test_cortex_integration.py
```

## 5. Troubleshooting

- **Database Locks:** If ChromaDB is locked, ensure the MCP server process is not stuck.
- **Import Errors:** Ensure `PYTHONPATH` includes the project root.
- **Empty Results:** Check `cortex_get_stats` to ensure documents are indexed.

