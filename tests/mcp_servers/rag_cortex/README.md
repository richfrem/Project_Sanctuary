# RAG Cortex MCP Tests

## Testing Pyramid (per ADR 048)

| Layer | Description | Requires | Run Command |
|-------|-------------|----------|-------------|
| **1. Unit** | Fast, isolated, mocked | Nothing | `pytest tests/mcp_servers/rag_cortex/ -v -m "not integration"` |
| **2. Integration** | Real ChromaDB | `sanctuary-vector-db` container | `pytest tests/mcp_servers/rag_cortex/ -v -m integration` |
| **3. MCP Ops** | Tool interface | Server running | Via Antigravity/Claude Desktop |

---

## Prerequisites for Integration Tests

```bash
# Start ChromaDB container
podman compose up -d vector-db

# Verify it's running
curl -I http://localhost:8000/api/v2/heartbeat
```

---

## Test Files by Layer

### Layer 1: Unit Tests (no dependencies)
- `test_models.py` - Data models
- `test_validator.py` - Input validation
- `test_cache_operations.py` - Mnemonic cache
- `test_enhanced_diagnostics.py` - Error handling

### Layer 2: Integration Tests (real ChromaDB)
- `test_integration_real_db.py` - Full ingestion/query flow
- `test_operations.py` (marked `@pytest.mark.integration`)
- `test_cortex_ingestion.py` (marked `@pytest.mark.integration`)

### Standalone Integration Script
```bash
python3 mcp_servers/rag_cortex/run_cortex_integration.py --run-full-ingest
```

---

## CI/CD Behavior

- **Unit tests**: Always run
- **Integration tests**: Auto-skipped if ChromaDB unavailable (see `conftest.py`)
