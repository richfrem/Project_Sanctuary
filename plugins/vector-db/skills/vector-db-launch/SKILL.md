---
name: vector-db-launch
<<<<<<< HEAD
description: Start the Native Python ChromaDB background server. Use when semantic search returns connection refused on port 8110, or when the user wants to enable concurrent agent read/writes.
---

# Vector DB Launch (Python Native Server)

ChromaDB provides the vector database backend for semantic search. If configured for Option C (Native Server) in `vector_profiles.json`, the database must be running as a background HTTP service to be accessed by `operations.py`.

## When You Need This

- **RAG ingest fails** with connection refused to `127.0.0.1:8110`
- **Semantic search** hangs or fails to connect
- The user has explicitly selected **Option 2 (Python Native Server)** during `vector-db-init`
=======
description: Start and verify the ChromaDB vector database container via Podman. Use when RAG ingestion fails, semantic search returns no results, or the Cortex MCP tools report connection errors to ChromaDB. Handles Podman machine startup, stale container cleanup, and health verification.
---

# Vector DB Launch (ChromaDB via Podman)

ChromaDB provides the vector database backend for semantic search (RAG Cortex).

## When You Need This

- **RAG ingest fails** with `CortexOperations is None` or connection errors
- **Semantic search** returns empty results or connection refused on port 8110
- **Protocol 128 Phase IX** (Ingestion & Closure) needs ChromaDB running
>>>>>>> origin/main

## Pre-Flight Check

```bash
# Check if ChromaDB is already running
<<<<<<< HEAD
curl -sf http://127.0.0.1:8110/api/v1/heartbeat > /dev/null && echo "✅ ChromaDB running" || echo "❌ ChromaDB not running"
```

If it prints "✅ ChromaDB running", you're done. If not, proceed.

## Launching the Server

The server must be launched using the `chroma run` command, bound to the data path defined in the profile's `chroma_data_path` field.

**CRITICAL INSTRUCTION FOR AGENT:**
The `chroma run` command blocks the terminal. You **must** instruct the user to run this command in a NEW, SEPARATE terminal window, or use `nohup` / `&` to background the process. 

Do not try to run a blocking `chroma run` loop inside your own execution context, or you will freeze.

### Step 1: Run the Server Command
Instruct the user to execute the following command in their terminal (from the project root):

```bash
chroma run --host 127.0.0.1 --port 8110 --path .knowledge_vector_data
```

### Step 2: Verify Connection
After the user confirms the server is running, verify it via API:

```bash
curl -sf http://127.0.0.1:8110/api/v1/heartbeat
```

It should return a JSON response containing a timestamp `{"nanosecond heartbeat": ...}`.

---
=======
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
>>>>>>> origin/main

## Troubleshooting

| Symptom | Fix |
|---------|-----|
<<<<<<< HEAD
| `chroma: command not found` | The user hasn't run the `vector-db-init` skill yet. Run it to `pip install chromadb`. |
| Port 8110 already in use | Another process (or zombie chroma process) is using the port. `lsof -i :8110` to find and kill it. |
| Permission Denied for data directory | Ensure the user has write access to the `.knowledge_vector_data` directory. |

## Alternative: In-Process Mode
If the user decides they do not want to run a background server, you can instruct them to set `chroma_host` to an empty string `""` in their profile in `.agent/learning/vector_profiles.json`. 

The `operations.py` library will automatically fallback to "Option A" (`PersistentClient`) and initialize the database locally inside the python process without needing this skill.
=======
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
>>>>>>> origin/main
