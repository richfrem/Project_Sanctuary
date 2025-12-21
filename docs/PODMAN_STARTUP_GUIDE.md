# Podman Startup Guide - Project Sanctuary

**Quick Reference:** Unified orchestration for the "Fleet of 8" MCP infrastructure.

---

## The Fleet of 8 Architecture
Project Sanctuary now operates a consolidated **Fleet of 8** containers, managed as a single stack via `docker-compose.yml`. This architecture ensures all internal services (Domain, Cortex, Git, etc.) can communicate over a private network while exposing standard ports to the host.

### 1. The Core Stack (project_sanctuary)
These 8 containers are defined in the root [docker-compose.yml](../../docker-compose.yml):

| Container Name | Role | Port (Host) | Description |
|----------------|------|-------------|-------------|
| `sanctuary_utils` | Utils | 8100 | Time, Math, Calculator, UUID tools. |
| `sanctuary_filesystem` | Code/FS | 8101 | File operations, code analysis, grep. |
| `sanctuary_network` | Network | 8102 | Brave Search, Fetch, HTTP clients. |
| `sanctuary_git` | Git | 8103 | Isolated Git workflow operations. |
| `sanctuary_cortex` | Cortex | 8104 | RAG server (connects to Vector DB & Ollama). |
| `sanctuary_domain` | Domain | 8105 | Chronicle, Task, ADR, Persona management. |
| `sanctuary_vector_db` | Vector DB | 8110 | ChromaDB backend storage. |
| `sanctuary_ollama_mcp` | Ollama | 11434 | Ollama LLM / Embedding compute backend. |

### 2. Infrastructure Services (External)
- **`mcpgateway`**: The IBM ContextForge Gateway (Port 4444).
- **`helloworld-mcp`**: Demo tool for gateway validation (Port 8005).

---

## Quick Start: Deployment Workflow (ADR-065)

### 1. Build & Start the Fleet
Use the unified Makefile for all operations:
```bash
# Build and start all 8 containers
make up

# Force rebuild (useful after code changes)
make up force=true
```

### 2. Verify Container Health
```bash
# Check status of the fleet
make status

# Or manually:
podman ps --filter "name=sanctuary"
```

### 3. Verify Gateway Registration & Tools
Use the **Gateway Client CLI** to verify tools are federated:
```bash
# Check all 84 tools across 6 servers
python3 -m mcp_servers.gateway.gateway_client tools -v

# Check specific server
python3 -m mcp_servers.gateway.gateway_client tools --server sanctuary_git

# Check registered servers
python3 -m mcp_servers.gateway.gateway_client servers
```

---

## Managing Individual Services
While the stack is managed as a whole, you can interact with specific containers:

```bash
# Restart a specific service
podman compose restart sanctuary_cortex

# View logs for a specific service
podman compose logs -f sanctuary_git

# Stop the entire fleet
podman compose down
```

---

## Pre-requisites & Environment
- **Gateway**: Must be running (usually via the sibling `sanctuary-gateway` repo).
- **Environment**: Your `.env` must define `MCPGATE_BEARER_TOKEN` for `fleet_orchestrator.py` to register tools.
- **SSL**: The gateway uses HTTPS (Port 4444) by default.

---

## Documentation Links
- **[Gateway README](../mcp_servers/gateway/README.md)**: Details on the 3-Layer Fleet Pattern.
- **[Architecture (ADR 060)](../ADRs/060_gateway_integration_patterns.md)**: Design rationale for the Fleet of 7/8.
- **[Fleet Spec](../mcp_servers/gateway/fleet_spec.py)**: Canonical definitions for all fleet servers.
