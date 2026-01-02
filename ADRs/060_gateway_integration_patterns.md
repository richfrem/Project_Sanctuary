# Gateway Integration Patterns - Hybrid Fleet

**Status:** ✅ **ACCEPTED**  
**Date:** 2025-12-17  
**Author:** Antigravity AI  
**Red Team Reviewers:** Claude 4.5 ✅, GPT 5.2 ✅, Gemini 3 ✅, Grok 4 ✅  
**Final Verdict:** APPROVED WITH MANDATORY GUARDRAILS  
**Last Updated:** 2025-12-17 (Final - Grok 4 RAG Frontend Gap Fix)

> [!NOTE]
> **Red Team Review Complete.** All four AI reviewers approved this architecture with mandatory guardrails.
> 
> **Grok 4 Critical Finding:** Modified based on Grok 4's identification of the "RAG Frontend Gap" - `sanctuary_vector_db` and `sanctuary_ollama` are backends, NOT MCP servers. Added `sanctuary_cortex` as the actual MCP server layer.


---

## Context

With the Gateway successfully deployed as an external service (ADR 058), we must decide how to connect Project Sanctuary's 10 script-based MCP servers without violating the decoupling mandate.

**Current State:**
- 2/12 servers containerized: `sanctuary_vector_db`, `sanctuary_ollama`
- 10/12 servers run as Python scripts via stdio
- Gateway is a black box at `https://localhost:4444`

**Requirements:**
- Maintain ADR 058 isolation (no monolithic dependencies)
- Support 100+ servers long-term
- Clear security boundaries
- Scalable architecture
- **Avoid "Orchestration Fatigue"** (Red Team feedback)

**Red Team Analysis:**
Three integration patterns were evaluated. The pure Fleet approach was modified based on Red Team feedback identifying "Orchestration Fatigue" as a critical flaw.

---

## Pattern Analysis

### Pattern A: The "Trojan Horse" (Volume Mounting)
**Mechanism:** Mount `./mcp_servers` into the Gateway container.

**Red Team Verdict:** ❌ **REJECT**

**Security Issues:**
- Violates ADR 058 decoupling mandate
- Creates "Dependency Hell" (Gateway must install dependencies for 12+ different tools)
- Tight coupling between Gateway and Project Sanctuary
- Version conflicts inevitable
- Single point of failure

---

### Pattern B: The "Bridge" (Stdio over SSH/Docker Exec)
**Mechanism:** Gateway uses specialized client to execute commands on host OS.

**Red Team Verdict:** ⚠️ **RISKY**

**Security Issues:**
- Requires shell access from Gateway to host (major security risk)
- Complex SSH key management
- Difficult to audit
- Breaks container isolation model
- Hard to debug

---

### Pattern C: The "Hybrid Fleet" (Containerized Service Groups)
**Mechanism:** Consolidate MCP servers into 4 logical clusters based on risk profile. Each cluster runs as a container exposing SSE endpoints.

**Red Team Verdict:** ✅ **APPROVED WITH MODIFICATION**

**Evolution from Pure Fleet:**
- **Original:** 12 separate containers (1 per tool)
- **Problem:** "Orchestration Fatigue" - unsustainable for project size
- **Solution:** Consolidate into 4 logical service groups

---

## Decision (Red Team Approved)

> [!NOTE]
> **Final Decision:** Approved unanimously by Red Team (Claude 4.5, GPT 5.2, Gemini 3, Grok 4) with mandatory guardrails.

**ACCEPTED:** Adopt **Pattern C: The Hybrid Fleet (Cluster Strategy)** - consolidating 10 script-based servers into a **Fleet of 7 Physical Containers** organized as **5 Logical Clusters**.

### Fleet of 7: Physical Container Inventory

| # | Container Name | Type | Logical Cluster | Role | Tools/Services |
|---|---------------|------|-----------------|------|----------------|
| 1 | `sanctuary_utils` | **NEW** | Utils (Low Risk) | MCP Server | Time, Calculator, UUID, String |
| 2 | `sanctuary_filesystem` | **NEW** | Filesystem (Privileged) | MCP Server | File Ops, Grep, Patch, Code |
| 3 | `sanctuary_network` | **NEW** | Network (External) | MCP Server | Brave Search, Fetch, HTTP |
| 4 | `sanctuary_git` | **NEW** | Git (Dual-Permission) | MCP Server | Git Workflow |
| 5a | `sanctuary_cortex` | **NEW** | Intelligence (Heavy) | MCP Server | RAG Query, Ingest, Cache |
| 5b | `sanctuary_vector_db` | **EXISTING** | Intelligence (Heavy) | Backend Storage | ChromaDB |
| 5c | `sanctuary_ollama` | **EXISTING** | Intelligence (Heavy) | Backend Compute | Ollama LLM |

> [!IMPORTANT]
> **Grok 4's "RAG Frontend Gap" Fix:** `sanctuary_vector_db` and `sanctuary_ollama` are **backends** (storage & compute). They do NOT expose MCP tools. `sanctuary_cortex` is the **MCP Server** that connects to these backends and exposes the tools via SSE.

**Summary:**
- **5 NEW containers** to build (including `sanctuary_cortex`)
- **2 EXISTING containers** unchanged (backends only)
- **= 7 TOTAL physical containers**
- **Organized into 5 Logical Clusters** (Intelligence cluster has 3 containers: 1 MCP server + 2 backends)

### Red Team Voting Summary

| Option | Physical Containers | Security | Orchestration | Claude | GPT | Gemini | Grok | Final |
|--------|-------------------|----------|---------------|--------|-----|--------|------|-------|
| Option A (Pure Fleet) | 12 | ✅ Perfect | ❌ Fatigue | ❌ | ❌ | ❌ | ❌ | ❌ REJECT |
| Option B (Rule of 4) | 4 | ❌ Git boundary | ✅ Simple | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ RISKY |
| **Option C (Fleet of 7)** | 7 | ✅ Strict | ✅ Manageable | ✅ | ✅ | ✅ | ✅ | ✅ **APPROVED** |

---

## Option Analysis: Container Consolidation Strategy

**Current State:**
- ✅ 2 containers **already containerized** (no changes needed):
  - `sanctuary_vector_db` (ChromaDB)
  - `sanctuary_ollama` (Ollama)
- ⏳ 10 script-based MCP servers (need containerization)

**Options Evaluated:**

### Option A: Pure Fleet (1 Container Per Tool)
**Strategy:** Create 10 new containers (1 per script-based server)

**Total Containers:** 12 (2 existing + 10 new)

**Pros:**
- Maximum isolation (each tool is independent)
- Simple failure domain (1 tool = 1 container)
- Easy to understand

**Cons:**
- ❌ **Orchestration Fatigue** (managing 12 containers)
- ❌ 12 separate Dockerfiles to maintain
- ❌ 12 separate health checks
- ❌ 12 separate log streams
- ❌ Complex docker-compose.yml

**Red Team Verdict:** ❌ **REJECT** (Orchestration Fatigue)

---

### Option B: Rule of 4 (Risk-Based Clustering)
**Strategy:** Consolidate 10 servers into 2 new clusters

**Total Containers:** 4 (2 existing + 2 new)
- `sanctuary_utils` (time, calculator, uuid, string)
- `sanctuary_network` (brave, fetch, git, http)

**Pros:**
- ✅ Reduces orchestration complexity (12 → 4)
- ✅ Risk-based grouping
- ✅ Manageable for project size

**Cons:**
- ⚠️ **Security Boundary Violation:** Git needs BOTH network + filesystem
- ⚠️ Blast radius (if utils crashes, 5-6 tools fail)

**Red Team Verdict:** ⚠️ **RISKY** (Security boundary violation)

---

### Option C: Rule of 5 (Revised Clustering) ✅ RECOMMENDED
**Strategy:** Consolidate 10 servers into 3 new clusters, keep 2 existing

**Total Containers:** 5 (2 existing + 3 new)
- **Existing (unchanged):**
  - `sanctuary_vector_db` (ChromaDB)
  - `sanctuary_ollama` (Ollama)
- **New:**
  - `sanctuary_utils` (time, calculator, uuid, string)
  - `sanctuary_filesystem` (file ops, grep, patch, code)
  - `sanctuary_network` (brave, fetch, http)
  - `sanctuary_git` (git only - isolated due to dual permissions)

**Pros:**
- ✅ Reduces orchestration complexity (12 → 5)
- ✅ Maintains strict security boundaries
- ✅ Git isolated (network + filesystem permissions)
- ✅ Existing containers unchanged (no migration risk)
- ✅ Manageable for project size

**Cons:**
- ⚠️ Slightly more complex than Rule of 4 (5 vs 4)
- ⚠️ Blast radius still exists (mitigated with process isolation)

**Red Team Verdict:** ✅ **APPROVED**

---

### Decision: Adopt Option C (Fleet of 7)

**Consolidation Summary:**
```
10 script-based servers → 5 new containers (58% reduction vs Pure Fleet)
+ 2 existing containers (backends unchanged)
= 7 total physical containers
→ Organized into 5 logical clusters
```

**Comparison:**
| Metric | Option A (Pure) | Option B (Rule of 4) | Option C (Fleet of 7) |
|--------|----------------|---------------------|----------------------|
| Total Containers | 12 | 4 | 7 |
| New Containers | 10 | 2 | 5 |
| Orchestration Complexity | 🔴 HIGH | 🟢 LOW | 🟢 LOW |
| Security Boundaries | 🟢 PERFECT | 🔴 VIOLATED | 🟢 STRICT |
| Blast Radius | 🟢 MINIMAL | 🔴 HIGH | 🟡 MEDIUM |
| Maintainability | 🔴 POOR | 🟢 GOOD | 🟢 GOOD |

---

### The "Fleet of 7" Cluster Strategy

Instead of managing 12 separate containers, we consolidate MCP servers into 5 logical clusters (7 physical containers) based on **risk profile** and **dependency similarity**:

---

### 1. `sanctuary_utils` (The "Low Risk" Cluster)
**Tools:** Time, Math, Calculator, UUID, String utilities

**Architecture:**
- Single lightweight Python/FastAPI container
- One SSE endpoint (`http://sanctuary_utils:8000/sse`)
- Internal routing to multiple tool implementations
- No external dependencies, no secrets, no file system access

**Benefit:** Eliminates 5-6 containers instantly while maintaining isolation from risky operations.

**Example:**
```python
# Single container serving multiple safe tools
@app.post("/sse")
async def handle_tool_call(request: ToolRequest):
    if request.tool == "time":
        return get_current_time()
    elif request.tool == "calculator":
        return calculate(request.params)
    # ... more safe tools
```

---

### 2. `sanctuary_filesystem` (The "Privileged" Cluster)
**Tools:** File System, grep, patch, code analysis

**Architecture:**
- Dedicated container with **controlled volume mounts**
- Restricted to specific directories (e.g., `/workspace:/app/workspace:ro`)
- **No network access** (isolated from external threats)
- Separate security boundary from "safe" utils
- Audit logging for all file operations

**Security Rationale:** File system access requires distinct isolation from network and compute tools.

---

### 3. `sanctuary_network` (The "External" Cluster)
**Tools:** Brave Search, Fetch, HTTP clients

**Architecture:**
- Container with **outbound network access**
- Injected secrets (API keys) via environment variables
- **No file system access** beyond temp directories
- Rate limiting and request logging

**Security Rationale:** Network tools need internet access but should be isolated from file system.

**Note:** Git is NOT in this cluster (see #4).

---

### 4. `sanctuary_git` (The "Isolated Dual-Permission" Cluster)
**Tools:** Git Workflow

**Architecture:**
- **Dedicated isolated container** (not grouped with other tools)
- Requires **both** network access (GitHub) AND filesystem access (clone/commit)
- Controlled volume mounts for repository access
- SSH key management for private repos
- Audit logging for all git operations

**Security Rationale:** Git's unique dual-permission requirement (network + filesystem) necessitates complete isolation from other tools to prevent privilege escalation attacks.

**Why Isolated:**
- If grouped with `sanctuary_network`: Could exploit network access to exfiltrate files
- If grouped with `sanctuary_filesystem`: Could exploit filesystem to leak secrets
- **Solution:** Complete isolation in dedicated container

---

### 5. `sanctuary-intelligence` (The "Heavy" Cluster)
**Tools:** RAG Cortex, Memory, Embeddings

**Architecture:**
- **Already handled** by existing `sanctuary_vector_db` and `sanctuary_ollama`
- No changes needed (these are already containerized)
- Existing security boundaries maintained

---

## Fleet of 7 Architecture Diagram

![mcp_gateway_fleet_detailed](docs/architecture_diagrams/system/mcp_gateway_fleet_detailed.png)

*[Source: mcp_gateway_fleet_detailed.mmd](docs/architecture_diagrams/system/mcp_gateway_fleet_detailed.mmd)*


**Legend:**
- 🔵 **Blue (Client):** MCP clients connecting via HTTPS
- 🟠 **Orange (Gateway):** Stateless HTTP proxy (IBM ContextForge)
- 🟢 **Green (#1 Utils):** Low risk, pure compute, no I/O - **NEW**
- 🟠 **Orange (#2 Filesystem):** Privileged, file system access, no network - **NEW**
- 🟣 **Purple (#3 Network):** External, outbound network, no filesystem - **NEW**
- 🔴 **Red (#4 Git):** Dual permissions (filesystem + network), isolated - **NEW**
- 🔵 **Teal (#5 Intelligence Cluster):**
  - **5a Cortex:** MCP Server (brain) - **NEW**
  - **5b VectorDB:** Backend storage - **EXISTING**
  - **5c Ollama:** Backend compute - **EXISTING**

---

## Mandatory Guardrails (6 Required)

> [!CAUTION]
> **These 6 guardrails are MANDATORY.** Implementation cannot proceed without them.
> 
> - **1-3:** Original (Fault Containment, Self-Registration, Network Addressing)
> - **4:** Resource Caps (Grok 4)
> - **5-6:** Grok 4 Nits (Volumes, Chain Timeouts)

### Guardrail 1: Intra-Cluster Fault Containment

**Requirement:** A single tool crash (e.g., "Math" division by zero) must NOT crash the entire container.

**Implementation:**
```python
# Entry point must handle exceptions gracefully
@app.post("/sse")
async def handle_tool_call(request: ToolRequest):
    try:
        if request.tool == "calculator":
            return await calculator_service(request.params)
        elif request.tool == "time":
            return await time_service(request.params)
    except Exception as e:
        # Container stays up, only this tool call fails
        logger.error(f"Tool {request.tool} failed: {e}")
        return {"error": str(e), "tool": request.tool, "status": "tool_failure"}
    # Other tools continue working
```

**Rationale:** Blast radius containment - one faulty tool should not take down sibling tools.

---

### Guardrail 2: Dynamic Self-Registration

**Requirement:** To prevent "Registry Drift," containers MUST POST their tool manifests to the Gateway on startup.

**Implementation:**
```python
# On container startup
async def register_with_gateway():
    manifest = {
        "server_name": "sanctuary_utils",
        "endpoint": "http://sanctuary_utils:8000/sse",
        "tools": ["time", "calculator", "uuid", "string"],
        "version": "1.0.0",
        "health_check": "/health"
    }
    response = await httpx.post(
        f"{GATEWAY_URL}/api/servers/register",
        json=manifest,
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    logger.info(f"Registered with Gateway: {response.status_code}")
```

**Rationale:** Self-healing system. No duplicate config files. Single source of truth.

**ChatGPT Commentary:** *"This is the secret weapon. Instead of editing Gateway config every time you add a tool, the tool tells the Gateway it exists."*

---

### Guardrail 3: Network Addressing

**Requirement:** All inter-container communication MUST use Docker Network Aliases.

**Correct:**
```python
GATEWAY_URL = "http://sanctuary-gateway:4444"
VECTOR_DB_URL = "http://sanctuary_vector_db:8000"
```

**WRONG:**
```python
# NEVER use these!
GATEWAY_URL = "http://localhost:4444"      # ❌ Won't work in containers
GATEWAY_URL = "http://192.168.1.100:4444"  # ❌ Hardcoded IP breaks portability
```

**Rationale:** Docker network aliases are stable, portable, and DNS-resolved. Localhost and hardcoded IPs break in containerized environments.

---

### Guardrail 4: Resource Caps (Grok 4 Requirement)

**Requirement:** All `docker-compose` service definitions MUST include `deploy.resources.limits` to prevent the 7 containers from overwhelming the dev machine.

**Implementation:**
```yaml
# docker-compose.yml
services:
  sanctuary_utils:
    build: ./mcp_servers/utils
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: '256M'
        reservations:
          cpus: '0.25'
          memory: '128M'
  
  sanctuary_cortex:
    build: ./mcp_servers/rag_cortex
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: '1G'  # Higher for RAG operations
        reservations:
          cpus: '0.5'
          memory: '512M'
```

**Recommended Limits:**
| Container | CPU Limit | Memory Limit | Rationale |
|-----------|-----------|--------------|-----------|
| sanctuary_utils | 0.5 | 256M | Lightweight, stateless |
| sanctuary_filesystem | 0.5 | 256M | I/O bound, not compute |
| sanctuary_network | 0.5 | 256M | HTTP client, low memory |
| sanctuary_git | 0.5 | 512M | Git operations need memory |
| sanctuary_cortex | 1.0 | 1G | RAG embedding, heavier |
| sanctuary_vector_db | 1.0 | 2G | Database operations |
| sanctuary_ollama | 2.0 | 4G | LLM inference is heavy |

**Rationale:** 7 containers running without limits can exhaust system resources. Explicit caps prevent runaway processes and ensure fair resource sharing.

---

### Guardrail 5: Persistent Volumes (Grok 4 Nit)

**Requirement:** All containers with persistent data MUST define Podman volumes to prevent data loss on restarts.

**Implementation:**
```yaml
# docker-compose.yml
volumes:
  vector_db-data:
  ollama-models:

services:
  sanctuary_vector_db:
    volumes:
      - vector_db-data:/chroma/chroma  # Persist embeddings
  
  sanctuary_ollama:
    volumes:
      - ollama-models:/root/.ollama    # Persist downloaded models
```

**Rationale:** Without explicit volumes, data is lost when containers restart. ChromaDB embeddings and Ollama models are expensive to regenerate.

---

### Guardrail 6: Chain Timeouts (Grok 4 Nit)

**Requirement:** Multi-backend chains in `sanctuary_cortex` MUST enforce per-hop timeouts to prevent latency spikes.

**Implementation:**
```python
# In sanctuary_cortex router
import httpx

TIMEOUT_CONFIG = {
    "embedding": 5.0,     # VectorDB embedding lookup
    "search": 10.0,       # VectorDB similarity search
    "generate": 30.0,     # Ollama LLM generation (can be slow)
}

async def rag_query(query: str):
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG["embedding"]) as client:
        embeddings = await client.post(f"{VECTOR_DB_URL}/embed", json={"text": query})
    
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG["search"]) as client:
        results = await client.post(f"{VECTOR_DB_URL}/search", json={"embedding": embeddings})
    
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG["generate"]) as client:
        response = await client.post(f"{OLLAMA_URL}/generate", json={"prompt": results})
    
    return response
```

**Chain Timeout Limits:**
| Operation | Timeout | Rationale |
|-----------|---------|-----------|
| Embedding | 5s | Simple vector lookup |
| Search | 10s | Similarity computation |
| Generate | 30s | LLM can be slow |
| **Total Chain** | **45s max** | Cap end-to-end latency |

**Rationale:** Multi-hop RAG chains (embed → search → generate) can hit 100ms+ under load. Per-hop timeouts prevent cascading delays and ensure predictable response times.

---

## Implementation Plan


### Phase 1: Pilot Deployment (Week 1)
**Goal:** Deploy `sanctuary_utils` container

1. Create `mcp_servers/utils/Dockerfile`
2. Implement multi-tool SSE handler
3. Add to `docker-compose.yml`
4. Register with Gateway
5. **Success Criteria:** Gateway successfully calls "What time is it?"

### Phase 2: Privileged & Network (Week 2-3)
1. Deploy `sanctuary_filesystem` with volume mounts
2. Deploy `sanctuary_network` with API keys
3. Test cross-cluster isolation

### Phase 3: Integration (Week 4)
1. Register all 4 clusters with Gateway
2. Update Claude Desktop config
3. E2E testing across all tool categories

### Phase 4: Cutover (Week 5)
1. Deprecate direct stdio connections
2. Full Gateway routing
3. Monitor and optimize

---

## Developer Experience

### Hot Reloading Mandate

**Constraint:** All development Dockerfiles MUST support hot reloading.

**Rationale:** We cannot afford to rebuild containers for every code change during development.

**Implementation:**
```yaml
# docker-compose.dev.yml
services:
  sanctuary_utils:
    build:
      context: ./mcp_servers/utils
      target: development  # Multi-stage build
    volumes:
      - ./mcp_servers/utils:/app:ro  # Mount source code
    environment:
      - UVICORN_RELOAD=true
    command: uvicorn main:app --reload --host 0.0.0.0
```

**Mechanism:**
- Use `watchdog` or Uvicorn's `--reload` flag
- Volume-mount source code in `dev` compose profile
- Production builds use multi-stage Dockerfile (no source mounts)

**Example Dockerfile:**
```dockerfile
# Development stage (hot reload)
FROM python:3.11-slim AS development
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]

# Production stage (optimized)
FROM python:3.11-slim AS production
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

---

## Consequences

**Positive:**
- Maintains ADR 058 isolation principles (no dependency hell)
- **Reduces orchestration complexity** (4 containers vs 12)
- Clear security boundaries (risk-based clustering)
- Matches existing architecture (vector_db, ollama patterns)
- Enables independent cluster lifecycle management
- Supports heterogeneous runtimes (Python, Node.js, Rust, etc.)
- **Developer-friendly** (hot reloading mandate)

**Negative:**
- Requires containerizing 10 script-based servers (grouped into 3 new containers)
- Learning curve for Dockerfile creation
- Potential resource overhead (though reduced from 12 containers)
- Internal routing complexity within each cluster

**Risks:**
- Cluster sprawl if grouping logic is wrong
- Debugging across cluster boundaries
- Initial migration effort for existing scripts
- Blast radius if one cluster fails (affects multiple tools)

**Mitigation:**
- Use docker-compose for orchestration
- Implement health checks for all clusters
- Standardize Dockerfile templates
- Monitor cluster resource usage and split if needed
- Clear logging and tracing within clusters

---

## Side-by-Side Architecture & Port Strategy

To support legacy direct connections while migrating to the Gateway, we implement a **Side-by-Side** strategy:

**1. Dual-Mode Servers:**
All MCP servers (Utils, Network, Filesystem, Git, Cortex) are refactored to support two transport modes:
- **Legacy (Stdio):** Default when `PORT` is not set. Used by direct `python -m mcp_servers.xxx` calls.
- **Gateway (SSE):** Active when `PORT` is set. Used by Podman containers.

**2. Port Assignments:**
To prevent conflicts, Gateway-routed containers use distict host ports from the legacy defaults.

| Server | Legacy Port (Direct) | Gateway Host Port (Podman) | Container Internal |
|--------|----------------------|----------------------------|--------------------|
| Utils | N/A (Stdio) | 8100 | 8000 |
| Filesystem | N/A (Stdio) | 8101 | 8000 |
| Network | N/A (Stdio) | 8102 | 8000 |
| Git | N/A (Stdio) | 8103 | 8000 |
| Cortex | 8000/8004 | 8104 | 8000 |
| Domain | 8105 | 8105 | 8105 |
| vector_db | 8000 | 8110 | 8000 |

**3. Configuration Toggles:**
Clients choose their mode via config:
- `legacy_direct.json`: Uses Stdio command lines.
- `gateway_routed.json`: Uses Gateway URL (`http://localhost:4444/sse`) with Bearer Token.

---

## Related Documents

- [ADR 058: Decouple IBM Gateway to External Podman Service](058_decouple_ibm_gateway_to_external_podman_service.md)
- [Task 118: Red Team Analysis](../TASKS/backlog/118_red_team_analysis_gateway_server_connection_patter.md)
- [Task 119: Deploy Pilot - sanctuary_utils Container](../TASKS/backlog/119_deploy_pilot_sanctuary_utils_container.md) (to be created)
