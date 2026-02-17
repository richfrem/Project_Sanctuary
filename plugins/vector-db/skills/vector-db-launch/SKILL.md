---
name: vector-db-launch
description: Start and verify the ChromaDB vector database container via Podman. Use when RAG ingestion fails, semantic search returns no results, or the Cortex MCP tools report connection errors to ChromaDB. Handles Podman machine startup, stale container cleanup, and health verification.
---

# Vector DB Launch (ChromaDB via Podman)

ChromaDB provides the vector database backend for semantic search (RAG Cortex).

## When You Need This

- **RAG ingest fails** with `CortexOperations is None` or connection errors
- **Semantic search** returns empty results or connection refused on port 8110
- **Protocol 128 Phase IX** (Ingestion & Closure) needs ChromaDB running

## Pre-Flight Check

```bash
# Check if ChromaDB is already running
curl -sf http://localhost:8110/api/v2/heartbeat && echo "✅ ChromaDB running" || echo "❌ ChromaDB not running"
```

If running, you're done. If not, proceed.

## Step 1: Ensure Podman Machine is Running

```bash
# Check Podman machine state
podman machine inspect --format '{{.State}}' 2>/dev/null

# If "stopped", start it (takes ~15-20 seconds)
podman machine start

# Verify
podman machine inspect --format '{{.State}}'
# Expected: "running"
```

## Step 2: Start ChromaDB Container

```bash
# Try starting the container
podman compose -f docker-compose.yml up -d sanctuary_vector_db
```

### If "name already in use" Error

Stale container from a previous session. Remove and retry:

```bash
podman rm -f sanctuary_vector_db
podman compose -f docker-compose.yml up -d sanctuary_vector_db
```

## Step 3: Wait and Verify

ChromaDB takes ~10-15 seconds to be ready:

```bash
# Wait for health
echo "Waiting for ChromaDB..."
for i in $(seq 1 15); do
    curl -sf http://localhost:8110/api/v2/heartbeat > /dev/null 2>&1 && echo "✅ ChromaDB ready" && break
    sleep 2
done
```

## Architecture Notes

- **Image**: `chromadb/chroma:latest` (pre-built, no custom Dockerfile)
- **Port**: 8110 (host) → 8000 (container)
- **Data**: Persisted in Podman volume `sanctuary_chroma_data`
- **Network**: `mcp_network` bridge (shared with other fleet containers)
- **You do NOT need the full fleet** — `sanctuary_vector_db` runs independently

## Important: You Only Need This One Container

The old "Fleet of 8" architecture had many MCP containers. For local development with Protocol 128, you only need:
1. **Ollama** (host-native, not containerized) — for LLM inference
2. **`sanctuary_vector_db`** (Podman container) — for ChromaDB/RAG

The other 6 fleet containers (`sanctuary_cortex`, `sanctuary_utils`, etc.) are only needed if using the MCP Gateway architecture.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `podman: command not found` | Podman not installed — ask user |
| `podman machine` won't start | Try `podman machine rm` then `podman machine init` |
| Container starts but health fails | Check logs: `podman logs sanctuary_vector_db` |
| `mcp_network` not found | Create it: `podman network create mcp_network` |
| Image pull fails | `podman pull docker.io/chromadb/chroma:latest` |

## Ingestion Options

Once ChromaDB is running, you have multiple ingest strategies:

```bash
# Full ingest (rebuilds entire collection — slow, ~5-10 min)
python3 tools/cli.py ingest

# Incremental ingest (only files modified in last N hours — fast)
python3 tools/cli.py ingest --incremental --hours 1

# Incremental with no-purge (safest for mid-session updates)
python3 tools/cli.py ingest --no-purge --incremental --hours 2
```

**Prefer incremental** for mid-session updates. Use full ingest only when the collection is stale or corrupted.

## Integration Points

- **Protocol 128 Phase IX**: `tools/cli.py ingest` requires ChromaDB
- **`tools/cli.py snapshot --type seal`**: Seal works without ChromaDB (uses file-based cache), but ingestion needs it
- **Semantic search**: Any RAG query tool requires ChromaDB running
- **Reference**: [Podman Operations Guide](docs/operations/processes/PODMAN_OPERATIONS_GUIDE.md)
