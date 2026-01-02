# Protocol 122: Dynamic Server Binding

**Status:** CANONICAL
**Classification:** Infrastructure Standard
**Version:** 1.0
**Authority:** Project Sanctuary Core Team
**Linked Protocols:** 101, 114, 116, 125
---

# Protocol 122: Dynamic Server Binding

## Abstract

This protocol defines the standard for **Dynamic Server Binding** in Project Sanctuary's MCP Gateway Architecture, enabling late-binding tool discovery, centralized routing, and context-efficient scaling to 100+ MCP servers.

---

## 1. Motivation

**Problem:** Static 1-to-1 binding (1 config entry = 1 MCP server) creates:
- Context window saturation (8,400 tokens for 12 servers)
- Configuration complexity (180+ lines of manual JSON)
- Scalability limits (~20 servers maximum)
- Fragmented security policies
- No centralized audit trail

**Solution:** Dynamic Server Binding through a centralized MCP Gateway that:
- Reduces context overhead by 88% (8,400 → 1,000 tokens)
- Enables scaling to 100+ servers (5x increase)
- Centralizes security enforcement (Protocol 101)
- Provides unified audit logging
- Supports side-by-side deployment (zero-risk migration)

---

## 2. Architecture

### 2.1 Core Components

![MCP Dynamic Binding Architecture](../../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.png)

*[Source: mcp_dynamic_binding_flow.mmd](../../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.mmd)*

### 2.2 Service Registry Schema

**Registry Database:** SQLite (`registry.db`)

```sql
CREATE TABLE mcp_servers (
    name TEXT PRIMARY KEY,
    container_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    transport TEXT NOT NULL CHECK(transport IN ('stdio', 'http', 'sse')),
    capabilities JSON NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'stopped', 'error')),
    last_health_check TIMESTAMP,
    metadata JSON
);

CREATE TABLE tool_registry (
    tool_name TEXT PRIMARY KEY,
    server_name TEXT NOT NULL,
    description TEXT,
    parameters_schema JSON,
    read_only BOOLEAN DEFAULT TRUE,
    approval_required BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (server_name) REFERENCES mcp_servers(name)
);

CREATE INDEX idx_tool_server ON tool_registry(server_name);
CREATE INDEX idx_server_status ON mcp_servers(status);
```

**Example Registry Entry:**
```json
{
  "name": "rag_cortex",
  "container_name": "rag-cortex-mcp",
  "endpoint": "http://localhost:8001",
  "transport": "http",
  "capabilities": [
    "cortex_query",
    "cortex_ingest_full",
    "cortex_ingest_incremental",
    "cortex_get_stats",
    "cortex_cache_warmup"
  ],
  "status": "running",
  "last_health_check": "2025-12-15T10:45:00Z",
  "metadata": {
    "version": "1.0",
    "domain": "project_sanctuary.cognitive.cortex"
  }
}
```

---

## 3. Dynamic Binding Workflow

### 3.1 Tool Discovery (Startup)

```python
# Gateway startup sequence
async def initialize_gateway():
    # 1. Load registry
    registry = load_registry("config/registry.db")
    
    # 2. Discover all backend servers
    servers = registry.get_all_servers()
    
    # 3. Generate dynamic tool definitions
    tools = []
    for server in servers:
        for tool_name in server.capabilities:
            tool_def = await fetch_tool_definition(server, tool_name)
            tools.append(tool_def)
    
    # 4. Register tools with MCP
    mcp.register_tools(tools)
```

### 3.2 Tool Invocation (Runtime)

```python
# Gateway tool invocation
@mcp.tool()
async def cortex_query(query: str, max_results: int = 5) -> str:
    """Proxy tool - routes to backend server."""
    # 1. Lookup server for tool
    server = registry.get_server_for_tool("cortex_query")
    
    # 2. Validate allowlist (Protocol 101)
    allowlist.validate("cortex_query", {"query": query})
    
    # 3. Forward request to backend
    response = await proxy.call(server, "cortex_query", {
        "query": query,
        "max_results": max_results
    })
    
    # 4. Log invocation (audit trail)
    audit_log.record("cortex_query", query, response)
    
    return response
```

### 3.3 Request Flow Diagram

![mcp_dynamic_binding_flow](../../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.png)

*[Source: mcp_dynamic_binding_flow.mmd](../../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.mmd)*

---

## 4. Security Integration

### 4.1 Allowlist Format (`project_mcp.json`)

```json
{
  "version": "1.0",
  "project": "Project_Sanctuary",
  "allowlist": {
    "servers": [
      "rag_cortex",
      "task",
      "git_workflow",
      "protocol",
      "chronicle"
    ],
    "tools": {
      "git_workflow": [
        "git_get_status",
        "git_add",
        "git_smart_commit",
        "git_push_feature"
      ],
      "rag_cortex": [
        "cortex_query",
        "cortex_ingest_incremental"
      ]
    },
    "operations": {
      "git_smart_commit": {
        "approval_required": true,
        "reason": "Protocol 101 v3.0 enforcement"
      },
      "cortex_ingest_full": {
        "approval_required": true,
        "reason": "Database purge operation"
      }
    }
  }
}
```

### 4.2 Protocol 101 Integration

**Enforcement Points:**
1. **Tool Invocation** - Validate against allowlist before proxying
2. **Approval Workflow** - Trigger human approval for sensitive operations
3. **Audit Logging** - Record all tool invocations with arguments
4. **Rate Limiting** - Prevent abuse (future enhancement)

---

## 5. Transport Protocols

### 5.1 Supported Transports

| Transport | Use Case | Latency | Complexity |
|-----------|----------|---------|------------|
| **stdio** | Local development, MVP | Lowest | Lowest |
| **HTTP** | Production, containerized | Low | Medium |
| **SSE** | Streaming responses | Medium | High |
| **WebSocket** | Bidirectional (future) | Low | High |

### 5.2 Transport Selection

**MVP (Week 1-2):** stdio only (simplest)
```json
{
  "transport": "stdio",
  "command": "/path/to/.venv/bin/python",
  "args": ["-m", "mcp_servers.rag_cortex.server"]
}
```

**Production (Week 3-4):** HTTP for containerized backends
```json
{
  "transport": "http",
  "endpoint": "http://rag-cortex-mcp:8001",
  "health_check": "/health"
}
```

---

## 6. Health Checks & Resilience

### 6.1 Health Check Protocol

```python
# Gateway health monitoring
async def health_check_loop():
    while True:
        for server in registry.get_all_servers():
            try:
                # Ping server
                response = await http.get(f"{server.endpoint}/health")
                
                # Update registry
                if response.status == 200:
                    registry.update_status(server.name, "running")
                else:
                    registry.update_status(server.name, "error")
            except Exception as e:
                registry.update_status(server.name, "error")
                logger.error(f"Health check failed: {server.name}: {e}")
        
        await asyncio.sleep(30)  # Check every 30 seconds
```

### 6.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, server, tool_name, args):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen(f"Server {server} is unavailable")
        
        try:
            response = await proxy.call(server, tool_name, args)
            self.failure_count = 0
            self.state = "CLOSED"
            return response
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure = time.time()
            raise
```

---

## 7. Migration Strategy

### 7.1 Side-by-Side Deployment

**Phase 1: Add Gateway (Week 1)**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "🆕 Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sanctuary_gateway.server"]
    },
    "rag_cortex_legacy": {
      "displayName": "📦 RAG Cortex (Legacy)",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.rag_cortex.server"]
    }
    // ... other 11 legacy servers
  }
}
```

**Phase 2: Remove Legacy (Week 4)**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sanctuary_gateway.server"]
    }
    // Legacy servers removed
  }
}
```

### 7.2 Rollback Plan

**If Gateway Fails:**
1. Revert `claude_desktop_config.json` to backup
2. Restart Claude Desktop
3. All legacy servers still work
4. Recovery time: <5 minutes

---

## 8. Performance Specifications

### 8.1 Latency Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Registry lookup | <5ms (p99) | TBD |
| Gateway routing | <10ms (p95) | TBD |
| Proxy overhead | <15ms (p95) | TBD |
| End-to-end | <30ms (p95) | TBD |

### 8.2 Scalability Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Context overhead | 8,400 tokens | 1,000 tokens | 88% reduction |
| Max servers | ~20 | 100+ | 5x increase |
| Tools per server | 5-10 | 20+ | 2x increase |

---

## 9. Implementation

**Reference Implementation:** IBM ContextForge (Apache 2.0)
- Repository: https://github.com/IBM/mcp-context-forge
- Version: v0.9.0 (Nov 2025)
- Customization: Sanctuary allowlist plugin, Protocol 101/114 integration

**Timeline:** 4 weeks
- Week 1: Fork, deploy MVP (3 servers), evaluate
- Week 2-3: Customize (allowlist, Protocol 101/114)
- Week 4: Migrate all 12 servers, production deployment

---

## 10. Related Protocols

- **Protocol 101 v3.0:** Functional Coherence (test enforcement)
- **Protocol 114:** Guardian Wakeup (context initialization)
- **Protocol 116:** Ollama Container Network (containerized services)
- **Protocol 125:** Autonomous Learning (rapid tool integration)

---

## 11. References

- ADR 056: Adoption of Dynamic MCP Gateway Pattern
- ADR 057: Adoption of IBM ContextForge for Dynamic MCP Gateway
- Task 115: Design and Specify Dynamic MCP Gateway Architecture
- Task 116: Implement Dynamic MCP Gateway with IBM ContextForge
- Research: docs/mcp_gateway/research/ (13 documents)
- MCP Specification: https://modelcontextprotocol.io

---

**Status:** CANONICAL  
**Version:** 1.0  
**Effective Date:** 2025-12-15  
**Authority:** Project Sanctuary Core Team
