# Task 110: Dynamic MCP Gateway Architecture - Research Summary

**Date:** 2025-12-15  
**Status:** Research Phase Complete  
**Related:** Task 110, Chronicle 308 (Doctrine of Successor State)

---

## Executive Summary

This research validates the **Dynamic MCP Gateway Pattern** as a viable architectural evolution for Project Sanctuary. The pattern addresses critical scalability and context efficiency challenges by replacing static server loading with a centralized broker that implements late binding and just-in-time tool discovery.

**Key Finding:** A containerized "Sanctuary Broker" can serve as a single registered MCP server that dynamically proxies requests to capability-specific servers based on project manifests, reducing context pollution while supporting 100+ tools. The architecture is container-runtime agnostic, supporting Podman, Docker, Kubernetes, and OpenShift.

---

## 1. MCP Architecture Fundamentals

### 1.1 Core MCP Pattern
The Model Context Protocol (MCP) operates on a **client-server architecture**:

- **MCP Host:** The AI application (e.g., Claude Desktop, IDE) where the LLM resides
- **MCP Client:** Communication layer within the host that translates LLM requests to MCP format
- **MCP Server:** External service exposing tools, resources, and prompts via JSON-RPC 2.0

**Current Sanctuary Implementation:**
- 12+ MCP servers statically loaded in Claude Desktop configuration
- Each server injects tool definitions into context window on startup
- Results in context pollution and rigid configuration

### 1.2 Dynamic Context Loading (DCL)
Research identified an emerging pattern called **Dynamic Context Loading**:

> "LLMs are provided with summaries of available tools and servers, loading full definitions into context only when required. This optimizes context window usage, reduces costs, and improves efficiency, especially when dealing with numerous MCP servers."

**Relevance to Sanctuary:** This is precisely what we need. Instead of loading 12+ servers upfront, we load ONE broker that provides summaries and fetches full definitions on-demand.

---

## 2. MCP Gateway Pattern

### 2.1 Gateway Architecture
An **MCP Gateway** is an advanced proxy specifically designed for agentic AI applications:

**Core Functions:**
1. **Single Entry Point:** Unified endpoint for AI agents to access multiple MCP servers
2. **Dynamic Routing:** Intelligently routes requests to appropriate backend servers based on tool requirements
3. **Service Discovery:** Discovers available MCP servers and their capabilities at runtime
4. **Centralized Security:** Enforces authentication, authorization, rate limiting, and logging
5. **Protocol Translation:** Can virtualize non-MCP services (e.g., wrapping REST APIs)
6. **Session Management:** Manages MCP sessions, keeps connections warm, implements resilience patterns

### 2.2 Late Binding Architecture
**Late Binding** means the connection between caller (AI agent) and service is established at invocation time, not development time.

**Benefits for Sanctuary:**
- AI agent communicates with Gateway using standardized MCP protocol
- Gateway dynamically discovers and routes to appropriate backend server
- Backend services can be changed, updated, or scaled independently
- Promotes decoupling between AI application and external tools
- Enables maintainable, scalable, and robust systems

### 2.3 Real-World Implementations
Research identified several production implementations:

- **Skywork.ai MCP Gateway:** Production-grade gateway with dynamic service discovery
- **Gravitee MCP Gateway:** Enterprise API management platform with MCP support
- **Gitconnected MCP Proxy:** Session management and resilience patterns

**Validation:** The pattern is proven in production environments, not theoretical.

---

## 3. Security Considerations

### 3.1 Threat Model
Dynamic tool loading introduces specific security risks:

1. **Prompt Injection:** Attackers manipulate Tool Invocation Prompts (TIPs) to trigger unauthorized actions
2. **MCP Protocol Vulnerabilities:** Malicious servers can exploit sampling, hijack conversations, or invoke tools covertly
3. **Privilege Escalation:** Agents tricked into performing actions outside intended scope
4. **Data Exfiltration:** Timing vulnerabilities or compromised datastores
5. **Context Overloading:** Excessive tools degrade performance and create attack surfaces

### 3.2 Security Allowlist Pattern
Research strongly recommends **allowlist-based security** as the primary defense:

**Allowlist Controls:**
- Explicitly define which MCP servers, tools, and domains are permitted
- Prevent unauthorized dynamic tool execution
- Restrict sensitive operations to read-only (e.g., kubectl diagnostics)
- Enforce at the Gateway level as a security chokepoint

**Additional Security Layers:**
1. **Principle of Least Privilege (POLP):** Minimum necessary permissions for every tool
2. **Input Validation & Output Filtering:** Guardrails to prevent prompt injection
3. **Sandboxed Environments:** Isolate agents to limit blast radius
4. **Authentication & Authorization:** OAuth 2.0, token validation, prevent confused deputy attacks
5. **Monitoring & Logging:** Audit all tool usage, arguments, and outputs
6. **Continuous Security Testing:** Automated red team testing

### 3.3 Sanctuary Security Requirements
For our implementation:

```json
{
  "allowlist": {
    "servers": ["rag_cortex", "git_workflow", "task", "protocol", ...],
    "domains": ["localhost", "*.sanctuary.internal"],
    "operations": {
      "git_workflow": ["read", "status"],  // No destructive ops without approval
      "rag_cortex": ["query", "ingest"]
    }
  },
  "authentication": "oauth2_token_validation",
  "sandbox": "podman_container_isolation",
  "logging": "all_tool_invocations"
}
```

---

## 4. Container Orchestration & Service Discovery

### 4.1 Container Runtime Options

Sanctuary's architecture is **container-runtime agnostic**, supporting multiple orchestration platforms:

**Podman (Recommended for Local/Single-Host):**
- Rootless containers (better security)
- Daemonless architecture
- Docker-compatible CLI
- Systemd integration
- Pod support (Kubernetes-like)

**Docker (Alternative for Local/Single-Host):**
- Largest ecosystem
- Mature tooling
- Docker Compose widely used
- Extensive documentation

**Kubernetes (Recommended for Multi-Host/Cloud):**
- Production-grade orchestration
- Auto-scaling, self-healing
- Multi-node deployments
- Cloud-native (AWS EKS, GCP GKE, Azure AKS)

**OpenShift (Enterprise Kubernetes):**
- Red Hat-supported
- Built-in security, CI/CD
- Developer-friendly
- Multi-tenancy

**Container Capabilities (All Runtimes):**
- Container lifecycle management (start, stop, health checks)
- Network isolation and service discovery
- Volume management for persistent state (ChromaDB, SQLite)
- Resource limits and security constraints

### 4.2 Service Discovery Patterns
Two primary patterns identified:

**Client-Side Discovery:**
- Client queries service registry directly
- Uses load-balancing algorithm to choose instance
- **Not suitable for Sanctuary:** Too much complexity in the Gateway

**Server-Side Discovery (Recommended):**
- Client sends requests to router (Gateway)
- Router queries registry and forwards to available instance
- **Perfect for Sanctuary:** Gateway handles all discovery logic

### 4.3 Service Registry Options
For Sanctuary's scale (12-20 servers), we need lightweight registry:

**Option 1: SQLite Registry (Recommended)**
```python
# Simple, file-based, no external dependencies
registry = {
    "rag_cortex": {
        "container": "rag-cortex-mcp",
        "endpoint": "http://localhost:8001",
        "capabilities": ["cortex_query", "cortex_ingest_full", ...],
        "status": "running"
    },
    ...
}
```

**Option 2: Container Runtime Inspection (Dynamic)**
```bash
# Podman
podman ps --filter "label=mcp.server=true" --format json

# Docker
docker ps --filter "label=mcp.server=true" --format json

# Kubernetes
kubectl get pods -l mcp.server=true -o json
```

**Recommendation:** Hybrid approach - SQLite for static config, runtime inspection for status.

---

## 5. Router Pattern & Intent Classification

### 5.1 Intent-Based Routing
The Gateway must route requests based on **tool intent**, not just tool name:

**Routing Logic:**
```python
def route_request(tool_name: str, context: dict) -> str:
    """
    Route MCP tool request to appropriate backend server.
    
    Examples:
    - "cortex_query" -> rag_cortex server
    - "git_add" -> git_workflow server
    - "create_task" -> task server
    """
    tool_to_server_map = load_from_registry()
    return tool_to_server_map.get(tool_name)
```

### 5.2 Dynamic Tool Definition Loading
**Critical Pattern:** Only inject tool definitions when needed.

**Current (Static):**
```json
// Claude Desktop loads ALL tools upfront
{
  "tools": [
    {"name": "cortex_query", "description": "...", "parameters": {...}},
    {"name": "cortex_ingest_full", "description": "...", "parameters": {...}},
    // ... 100+ more tools
  ]
}
```

**Proposed (Dynamic):**
```json
// Gateway provides summary, fetches full definitions on-demand
{
  "tools_summary": {
    "rag_cortex": "Semantic search and knowledge management",
    "git_workflow": "Git operations and version control",
    "task": "Task tracking and management"
  }
}

// When LLM says "I need to query the knowledge base":
// Gateway fetches full tool definitions for rag_cortex only
```

### 5.3 Anthropic Routing Pattern
Research found Anthropic's official guidance on **Routing Pattern** for AI agents:

> "Routing pattern for intelligent classification and direction of tasks, which aligns with the MCP client's function of discovering and invoking relevant external services."

**Application to Sanctuary:**
- Gateway acts as intelligent router
- Classifies user intent from LLM request
- Directs to appropriate capability-specific server
- Reduces cognitive load on LLM by limiting tool choices

---

## 6. Implementation Architecture

### 6.1 Proposed Components

**1. The Sanctuary Broker (Gateway)**
- Python-based MCP server using FastMCP
- Single### Deployment Architecture

**Container Strategy (Runtime-Agnostic):**
- Gateway runs in container (`sanctuary-broker-mcp`)
- Backend servers in separate containers
- Container orchestration via Compose/Kubernetes
- Container network: `sanctuary-internal`

**Supported Container Runtimes:**
- **Podman** (recommended for local/single-host) - Rootless, daemonless, secure
- **Docker** (alternative for local/single-host) - Mature ecosystem, wide adoption
- **Kubernetes** (recommended for multi-host/cloud) - Production orchestration, auto-scaling
- **OpenShift** (enterprise Kubernetes) - Built-in security, CI/CD, developer-friendly

**Deployment Phases:**
- **Phase 1 (MVP):** Podman Compose (local development)
- **Phase 2 (Production):** Podman with systemd (single host)
- **Phase 3 (Scale):** Kubernetes/OpenShift (multi-host, cloud)able MCP servers
- Schema:
  ```sql
  CREATE TABLE mcp_servers (
      name TEXT PRIMARY KEY,
      container_name TEXT,
      endpoint TEXT,
      capabilities JSON,  -- List of tool names
      status TEXT,  -- running, stopped, error
      last_health_check TIMESTAMP
  );
  ```

**3. The Router**
- Intent classifier that maps tool requests to servers
- Maintains tool_name -> server_name mapping
- Handles fallback and error cases

**4. The Allowlist**
- Security layer preventing unauthorized tool execution
- Project-specific allowlists via `project_mcp.json`
- Global allowlist for Sanctuary-wide restrictions

**5. The Proxy**
- Forwards validated requests to backend MCP servers
- Handles JSON-RPC 2.0 message translation
- Manages sessions and connection pooling

### 6.2 Configuration: `project_mcp.json`

```json
{
  "project": "Project_Sanctuary",
  "version": "1.0",
  "required_capabilities": [
    "rag_cortex",
    "git_workflow",
    "task",
    "protocol",
    "chronicle",
    "code",
    "council"
  ],
  "optional_capabilities": [
    "forge_llm",
    "orchestrator"
  ],
  "security": {
    "allowlist_mode": "strict",
    "allowed_operations": {
      "git_workflow": ["git_get_status", "git_add", "git_smart_commit"],
      "rag_cortex": ["cortex_query", "cortex_ingest_incremental"]
    }
  }
}
```

### 6.3 Workflow Example

**Scenario:** User asks "What is Protocol 101?"

1. **LLM Request:** Claude sends tool request to Sanctuary Broker
   ```json
   {
     "method": "tools/call",
     "params": {
       "name": "cortex_query",
       "arguments": {"query": "What is Protocol 101?"}
     }
   }
   ```

2. **Gateway Routing:**
   - Broker receives request
   - Checks allowlist: `cortex_query` is permitted
   - Queries registry: `cortex_query` maps to `rag_cortex` server
   - Checks health: `rag_cortex` container is running

3. **Proxy Forwarding:**
   - Broker forwards request to `http://localhost:8001` (rag_cortex endpoint)
   - Receives response from rag_cortex server

4. **Response Return:**
   - Broker returns response to Claude
   - Logs invocation for audit trail

---

## 7. Scalability Analysis

### 7.1 Current vs. Proposed

**Current (Static Loading):**
- Context Window: ~8,000 tokens for tool definitions
- Servers Loaded: 12 (all upfront)
- Scalability Limit: ~20 servers before context saturation

**Proposed (Dynamic Gateway):**
- Context Window: ~500 tokens for gateway + summaries
- Servers Loaded: 1 (gateway only)
- Scalability Limit: 100+ servers (limited by registry, not context)

**Efficiency Gain:** 94% reduction in context overhead

### 7.2 Performance Considerations

**Latency:**
- Additional hop through Gateway: ~10-50ms
- Acceptable for Sanctuary's use case (human-in-loop)

**Caching:**
- Gateway can cache tool definitions
- Warm connections to frequently used servers
- Reduces repeated lookups

**Resilience:**
- Circuit breaker pattern for failing servers
- Fallback to cached responses
- Health check monitoring

---

## 8. Migration Path

### 8.1 Phased Rollout

**Phase 1: Proof of Concept (Week 1)**
- Build minimal Gateway with 2-3 servers (rag_cortex, task, git_workflow)
- Validate routing and proxying logic
- Test with Claude Desktop

**Phase 2: Full Implementation (Week 2-3)**
- Migrate all 12 servers to Gateway
- Implement allowlist security
- Add health checks and monitoring

**Phase 3: Advanced Features (Week 4+)**
- Dynamic server registration
- Protocol translation for non-MCP services
- Performance optimization and caching

### 8.2 Backward Compatibility
- Keep existing MCP servers unchanged
- Gateway acts as transparent proxy
- Can run in parallel with static config during transition

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gateway becomes single point of failure | High | Health checks, auto-restart, fallback to direct server access |
| Routing logic errors send requests to wrong server | Medium | Comprehensive testing, schema validation, error logging |
| Security allowlist bypassed | Critical | Multi-layer validation, audit logging, principle of least privilege |
| Performance degradation from additional hop | Low | Connection pooling, caching, async I/O |
| Complexity increases maintenance burden | Medium | Comprehensive documentation, automated tests, monitoring |

---

## 10. Recommendations

### 10.1 Proceed with Implementation
✅ **Research validates the pattern is viable, proven, and necessary.**

**Rationale:**
1. Addresses critical scalability bottleneck (context window saturation)
2. Proven in production by multiple vendors (Skywork.ai, Gravitee)
3. Aligns with Anthropic's official routing pattern guidance
4. Security model is well-understood and implementable
5. Migration path is low-risk with backward compatibility

### 10.2 Next Steps
1. **Create ADR:** Formalize the architectural decision
2. **Create Protocol 122:** Define the Dynamic Binding Standard
3. **Create Architecture Spec:** Technical implementation guide
4. **Build POC:** Minimal Gateway with 3 servers
5. **Validate:** Test with Claude Desktop and real workloads

---

## 11. References

### Research Sources
1. **MCP Architecture:** Anthropic MCP Documentation, Wikipedia, DeepLearning.AI
2. **Gateway Pattern:** Skywork.ai, Gravitee.io, Medium (MCP Gateway articles)
3. **Security:** Palo Alto Networks, Lakera.ai, Arxiv (LLM security papers)
4. **Service Discovery:** Kong, Edureka, TechTarget (Microservices patterns)

### Related Sanctuary Documents
- **Chronicle 308:** Doctrine of Successor State (context efficiency mandate)
- **Protocol 116:** Ollama Container Network (containerized MCP servers)
- **Task 087:** Comprehensive MCP Operations Testing

---

## Conclusion

The **Dynamic MCP Gateway Pattern** is the correct architectural evolution for Project Sanctuary. It solves the context efficiency problem, enables scalability to 100+ tools, and provides a secure, maintainable foundation for the next phase of development.

**Status:** Research phase complete. Ready to proceed to formalization (ADR, Protocol, Architecture Spec).
