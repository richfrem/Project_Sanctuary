# Podman Startup Guide - Project Sanctuary

**Quick Reference:** Essential Podman containers for MCP infrastructure

---

## Required Containers (Fleet of 7)

Project Sanctuary uses a **Fleet of 7** container architecture (see [ADR 060](../ADRs/060_gateway_integration_patterns__hybrid_fleet.md)):

**Currently Running (3 containers):**
1. **`sanctuary-vector-db`** - ChromaDB for RAG Cortex MCP (Backend)
2. **`sanctuary-ollama-mcp`** - Ollama for Forge LLM MCP (Backend)
3. **`sanctuary-gateway`** - IBM ContextForge Gateway for MCP routing (external repo)

**Planned (4 new containers per ADR 060):**
4. **`sanctuary-utils`** - Low-risk tools (time, calculator, UUID)
5. **`sanctuary-filesystem`** - File operations (grep, patch, code)
6. **`sanctuary-network`** - HTTP clients (brave search, fetch)
7. **`sanctuary-git`** - Git workflow (isolated dual-permission)
8. **`sanctuary-cortex`** - RAG MCP Server (connects to #1 and #2)

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
curl -I http://localhost:8000/api/v2/heartbeat

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
**Data:** `./.vector_data/` (persisted)  
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

### Sanctuary Gateway (MCP Router)

**⚠️ External Repository Required**

The gateway is managed in a separate repository: `sanctuary-gateway`

#### Initial Setup

1. **Clone the gateway repository** (sibling to Project_Sanctuary):
   ```bash
   cd ~/Projects
   git clone https://github.com/IBM/mcp-context-forge.git sanctuary-gateway
   cd sanctuary-gateway
   ```

2. **Build the container**:
   ```bash
   make container-build
   ```

3. **Generate SSL certificates**:
   ```bash
   make certs
   make certs-jwt
   ```

4. **Create `.env` file** in `sanctuary-gateway/`:
   ```bash
   # Database
   DATABASE_URL=sqlite:////app/data/mcp.db
   
   # JWT Keys (RS256)
   JWT_PUBLIC_KEY_PATH=/app/certs/jwt/public.pem
   JWT_PRIVATE_KEY_PATH=/app/certs/jwt/private.pem
   JWT_ALGORITHM=RS256
   
   # Admin credentials
   PLATFORM_ADMIN_EMAIL=your-email@example.com
   PLATFORM_ADMIN_PASSWORD=your-secure-password
   ```

5. **Start the gateway**:
   ```bash
   make podman-run-ssl
   ```

#### Daily Management

```bash
# Start gateway (from sanctuary-gateway repo)
cd ~/Projects/sanctuary-gateway
make podman-run-ssl

# Stop gateway
podman stop sanctuary-gateway

# View logs
podman logs sanctuary-gateway -f

# Restart gateway
podman restart sanctuary-gateway
```

**Port:** 4444 (HTTPS)  
**Data:** Podman volume `mcp_gateway_data` (persisted)  
**Admin UI:** https://localhost:4444/admin  
**Used by:** Future MCP tool routing (Task 117)

#### Verify Gateway Connection

From `Project_Sanctuary` repo:

1. **Create API token** at https://localhost:4444/admin/tokens
2. **Add to `.env`**:
   ```bash
   MCP_GATEWAY_ENABLED=true
   MCP_GATEWAY_URL=https://localhost:4444
   MCP_GATEWAY_VERIFY_SSL=false
   MCP_GATEWAY_API_TOKEN=<your-token-here>
   ```

3. **Run connectivity tests**:
   ```bash
   pytest tests/mcp_servers/gateway/integration/test_gateway_blackbox.py -v -m gateway
   ```

   Expected output:
   ```
   ✅ test_pulse_check PASSED       # Gateway is healthy
   ✅ test_circuit_breaker PASSED   # Security working
   ✅ test_handshake PASSED         # API token valid
   ```

**See:** [Gateway Setup Guide](mcp_gateway/README.md) for detailed configuration.

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
        response = requests.get("http://localhost:8000/api/v2/heartbeat")
        if response.status_code != 200:
            raise Exception("ChromaDB not healthy")
    except:
        logger.warning("ChromaDB not running, attempting to start...")
        subprocess.run(["podman", "compose", "up", "-d", "vector-db"])
        time.sleep(5)  # Wait for startup
```

**Recommendation:** Consider adding this as a future enhancement task.

---

## Environment Variables

Ensure `.env` file contains:

```bash
# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Ollama (for MCP infrastructure)
OLLAMA_HOST=http://ollama-model-mcp:11434
OLLAMA_MODEL=Sanctuary-Qwen2-7B:latest
```

**Note:** `OLLAMA_HOST` uses container network alias for MCP-to-MCP communication (Protocol 116).

---

## Quick Reference Card

| Service | Container Name | Port | Start Command | Health Check |
|---------|---------------|------|---------------|--------------|
| ChromaDB | `sanctuary-vector-db` | 8000 | `podman compose up -d vector-db` | `curl http://localhost:8000/api/v2/heartbeat` |
| Ollama | `sanctuary-ollama-mcp` | 11434 | `podman compose up -d ollama-model-mcp` | `curl http://localhost:11434/api/tags` |
| Gateway | `sanctuary-gateway` | 4444 | `make podman-run-ssl` (in gateway repo) | `curl -k https://localhost:4444/health` |

---

## Related Documentation

- [Gateway Setup Guide](mcp_gateway/README.md)
- [RAG Cortex SETUP.md](mcp/servers/rag_cortex/SETUP.md)
- [Forge LLM SETUP.md](mcp/servers/forge_llm/SETUP.md)
- [Protocol 116: Container Network Isolation](../01_PROTOCOLS/116_Container_Network_Isolation.md)
- [ADR 043: Containerize Ollama via Podman](../ADRs/043_containerize_ollama_model_service_via_podman.md)
- [ADR 058: Decouple IBM Gateway to External Podman Service](../ADRs/058_decouple_ibm_gateway_to_external_podman_service.md)
