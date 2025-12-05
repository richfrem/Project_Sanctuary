# Podman Startup Guide - Project Sanctuary

**Quick Reference:** Essential Podman containers for MCP infrastructure

---

## Required Containers

Project Sanctuary requires **2 Podman containers** for full MCP functionality:

1. **`sanctuary-vector-db`** - ChromaDB for RAG Cortex MCP
2. **`sanctuary-ollama-mcp`** - Ollama for Forge LLM MCP

---

## Quick Start (Recommended)

### Start All Services

```bash
# From project root
podman compose up -d vector-db ollama-model-mcp
```

**This single command starts both containers.**

### Verify Services

```bash
# Check container status
podman ps

# Expected output:
# CONTAINER ID  IMAGE                              STATUS
# <id>          chromadb/chroma:latest            Up X minutes
# <id>          ollama/ollama:latest              Up X minutes
```

### Test Services

```bash
# Test ChromaDB (should return 200)
curl -I http://localhost:8000/api/v1/heartbeat

# Test Ollama (should list models)
curl http://localhost:11434/api/tags
```

---

## Individual Service Management

### ChromaDB (RAG Cortex MCP)

```bash
# Start
podman compose up -d vector-db

# Stop
podman compose stop vector-db

# Restart
podman compose restart vector-db

# View logs
podman logs sanctuary-vector-db -f
```

**Port:** 8000  
**Data:** `./chroma_data/` (persisted)  
**Used by:** RAG Cortex MCP, Council MCP

---

### Ollama (Forge LLM MCP)

```bash
# Start
podman compose up -d ollama-model-mcp

# Stop
podman compose stop ollama-model-mcp

# Restart
podman compose restart ollama-model-mcp

# View logs
podman logs sanctuary-ollama-mcp -f
```

**Port:** 11434  
**Data:** `./ollama_models/` (persisted)  
**Model:** Sanctuary-Qwen2-7B:latest (auto-pulled on first start)  
**Used by:** Forge LLM MCP, Agent Persona MCP, Council MCP

---

## Daily Workflow

### Morning Startup

```bash
# Start all services
podman compose up -d

# Verify
podman ps
```

### Evening Shutdown

```bash
# Stop all services (preserves data)
podman compose stop

# Or stop specific services
podman compose stop vector-db ollama-model-mcp
```

### Check Service Health

```bash
# Quick health check
podman ps --filter "name=sanctuary"

# Detailed status
podman compose ps
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
podman logs sanctuary-vector-db
podman logs sanctuary-ollama-mcp

# Restart container
podman compose restart vector-db
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8000   # ChromaDB
lsof -i :11434  # Ollama

# Kill process or stop conflicting service
```

### GPU Not Detected (Ollama)

```bash
# Verify GPU access
podman exec sanctuary-ollama-mcp nvidia-smi

# If fails, check docker-compose.yml GPU configuration
```

### Data Persistence Issues

```bash
# Verify volumes
ls -la chroma_data/
ls -la ollama_models/

# If missing, containers will recreate on next start
```

---

## Advanced: Auto-Start on Boot

### macOS (Launch Agent)

Create `~/Library/LaunchAgents/com.sanctuary.podman.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.sanctuary.podman</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/podman</string>
        <string>compose</string>
        <string>up</string>
        <string>-d</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/Projects/Project_Sanctuary</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
```

**Load:**
```bash
launchctl load ~/Library/LaunchAgents/com.sanctuary.podman.plist
```

---

## MCP Server Auto-Start Integration

### Current Behavior
MCP servers **do NOT** auto-start Podman containers. They will fail if containers aren't running.

### Proposed Enhancement (Future)
Add health checks to MCP server startup:

```python
# Example: RAG Cortex MCP startup
def _ensure_chromadb_running():
    try:
        response = requests.get("http://localhost:8000/api/v1/heartbeat")
        if response.status_code != 200:
            raise Exception("ChromaDB not healthy")
    except:
        logger.warning("ChromaDB not running, attempting to start...")
        subprocess.run(["podman", "compose", "up", "-d", "vector-db"])
        time.sleep(5)  # Wait for startup
```

**Recommendation:** Create a task for this enhancement (T095?)

---

## Environment Variables

Ensure `.env` file contains:

```bash
# ChromaDB
CHROMA_HOST=http://localhost:8000

# Ollama (for MCP infrastructure)
OLLAMA_HOST=http://ollama-model-mcp:11434
OLLAMA_MODEL=Sanctuary-Qwen2-7B:latest
```

**Note:** `OLLAMA_HOST` uses container network alias for MCP-to-MCP communication (Protocol 116).

---

## Quick Reference Card

| Service | Container Name | Port | Start Command | Health Check |
|---------|---------------|------|---------------|--------------|
| ChromaDB | `sanctuary-vector-db` | 8000 | `podman compose up -d vector-db` | `curl http://localhost:8000/api/v1/heartbeat` |
| Ollama | `sanctuary-ollama-mcp` | 11434 | `podman compose up -d ollama-model-mcp` | `curl http://localhost:11434/api/tags` |

---

## Related Documentation

- [RAG Cortex SETUP.md](mcp/servers/rag_cortex/SETUP.md)
- [Forge LLM SETUP.md](mcp/servers/forge_llm/SETUP.md)
- [Protocol 116: Container Network Isolation](../01_PROTOCOLS/116_Container_Network_Isolation.md)
- [ADR 043: Containerize Ollama via Podman](../ADRs/043_containerize_ollama_model_service_via_podman.md)
