# Current State vs. Future State Architecture Analysis

**Research Date:** 2025-12-15  
**Task:** 110 - Dynamic MCP Gateway Architecture  
**Focus:** Internal architecture review, current implementation analysis, migration path

---

## Executive Summary

**Current State:** 12 independent MCP servers, each registered separately in Claude Desktop, using stdio transport, FastMCP framework.

**Future State:** 1 Gateway MCP server (Sanctuary Broker) that dynamically routes to 12 backend servers.

**Key Finding:** Our current architecture is **well-positioned** for Gateway migration. All servers use FastMCP, follow consistent patterns, and have clear domain boundaries.

---

## 1. Current Architecture Analysis

### 1.1 Server Inventory

**Total Servers:** 12  
**Framework:** FastMCP (Python)  
**Transport:** stdio (subprocess)  
**Registration:** Static in `claude_desktop_config.json`

| Server | Domain | Tools | Status |
|--------|--------|-------|--------|
| git_workflow | Git operations | 8 | ✅ Production |
| task | Task management | 6 | ✅ Production |
| adr | Architecture decisions | 5 | ✅ Production |
| chronicle | Historical records | 6 | ✅ Production |
| protocol | Protocol management | 5 | ✅ Production |
| rag_cortex | Knowledge base | 9 | ✅ Production |
| council | Multi-agent deliberation | 2 | ✅ Production |
| agent_persona | Persona dispatch | 5 | ✅ Production |
| forge_llm | Fine-tuned model queries | 2 | ✅ Production |
| config | Configuration management | 4 | ✅ Production |
| code | Code operations | 9 | ✅ Production |
| orchestrator | Strategic workflows | 2 | ✅ Production |

**Total Tools:** ~63 tools across 12 servers

### 1.2 Current Claude Desktop Configuration

**Format:** JSON configuration file  
**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Pattern (All Servers):**
```json
{
  "mcpServers": {
    "server_name": {
      "displayName": "Human Readable Name",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.server_name.server"],
      "env": {
        "PROJECT_ROOT": "/path/to/Project_Sanctuary",
        "PYTHONPATH": "/path/to/Project_Sanctuary",
        // Server-specific env vars
      },
      "cwd": "/path/to/Project_Sanctuary"
    }
  }
}
```

**Observations:**
1. **Consistent Pattern:** All servers follow same structure
2. **Environment Variables:** Each server gets `PROJECT_ROOT`, `PYTHONPATH`
3. **Working Directory:** All servers run from project root
4. **Subprocess Model:** Claude launches each as separate Python process

### 1.3 Current Server Implementation Pattern

**Example: RAG Cortex Server**

```python
# mcp_servers/rag_cortex/server.py
from fastmcp import FastMCP

# Initialize FastMCP with domain name
mcp = FastMCP("project_sanctuary.cognitive.cortex")

# Lazy initialization pattern
_cortex_ops = None

def get_ops():
    global _cortex_ops
    if _cortex_ops is None:
        from .operations import CortexOperations
        _cortex_ops = CortexOperations(PROJECT_ROOT)
    return _cortex_ops

# Tool registration with decorator
@mcp.tool()
def cortex_query(
    query: str,
    max_results: int = 5,
    use_cache: bool = False,
    reasoning_mode: bool = False
) -> str:
    """Perform semantic search query against the knowledge base."""
    try:
        validated = get_validator().validate_query(...)
        response = get_ops().query(...)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

# Server entry point
if __name__ == "__main__":
    mcp.run()  # Starts stdio server
```

**Pattern Analysis:**
- ✅ **FastMCP Framework:** All servers use FastMCP
- ✅ **Decorator-Based Tools:** `@mcp.tool()` for tool registration
- ✅ **JSON Responses:** All tools return JSON strings
- ✅ **Error Handling:** Try/catch with error JSON
- ✅ **Lazy Initialization:** Operations loaded on first use
- ✅ **Domain Naming:** Consistent `project_sanctuary.domain.subdomain`

### 1.4 Current Directory Structure

```
Project_Sanctuary/
├── mcp_servers/
│   ├── adr/
│   │   ├── server.py          # FastMCP server
│   │   ├── operations.py      # Business logic
│   │   ├── models.py          # Data models
│   │   └── validator.py       # Input validation
│   ├── chronicle/
│   │   ├── server.py
│   │   ├── operations.py
│   │   └── ...
│   ├── rag_cortex/
│   │   ├── server.py
│   │   ├── operations.py
│   │   ├── validator.py
│   │   └── ...
│   ├── git/
│   │   ├── server.py
│   │   ├── git_ops.py         # Git operations
│   │   └── ...
│   └── ... (12 servers total)
├── docs/architecture/mcp/
│   ├── architecture.md        # Current architecture doc
│   ├── servers/
│   │   ├── rag_cortex/README.md
│   │   ├── git/README.md
│   │   └── ...
│   └── ...
└── tests/mcp_servers/
    ├── rag_cortex/
    │   ├── test_cortex_operations.py
    │   └── integration/
    ├── git/
    │   └── test_git_ops.py
    └── ...
```

**Observations:**
- ✅ **Consistent Structure:** All servers follow same pattern
- ✅ **Separation of Concerns:** server.py (MCP), operations.py (logic), models.py (data)
- ✅ **Documentation:** Each server has README in docs/architecture/mcp/servers/
- ✅ **Testing:** Unit and integration tests per server

### 1.5 Current Context Overhead

**Measurement:** Tool definitions loaded into Claude's context window

**Estimated Token Count:**
```
Server: rag_cortex (9 tools)
- cortex_query: ~100 tokens
- cortex_ingest_full: ~120 tokens
- cortex_get_stats: ~80 tokens
- cortex_ingest_incremental: ~110 tokens
- cortex_cache_get: ~90 tokens
- cortex_cache_set: ~90 tokens
- cortex_cache_warmup: ~100 tokens
- cortex_guardian_wakeup: ~100 tokens
- cortex_cache_stats: ~80 tokens
Total: ~870 tokens

Server: git_workflow (8 tools)
- git_smart_commit: ~150 tokens
- git_get_safety_rules: ~200 tokens (large docstring)
- git_get_status: ~80 tokens
- git_add: ~100 tokens
- git_push_feature: ~120 tokens
- git_start_feature: ~130 tokens
- git_finish_feature: ~120 tokens
- git_diff: ~90 tokens
- git_log: ~80 tokens
Total: ~1,070 tokens

... (10 more servers)

TOTAL ESTIMATE: 8,000-10,000 tokens
```

**Problem:** This is 10-12% of Claude's 200K context window, consumed before any actual work.

---

## 2. Future State Architecture

### 2.1 Gateway Architecture

**New Structure:**
```
Claude Desktop
    ↓ (stdio)
Sanctuary Gateway (sanctuary-broker-mcp)
    ↓ (HTTP or stdio)
Backend MCP Servers (rag_cortex, git_workflow, task, ...)
```

**Gateway Responsibilities:**
1. **Single Entry Point:** One server registered in Claude Desktop
2. **Dynamic Routing:** Route tool calls to appropriate backend server
3. **Service Registry:** Maintain mapping of tools → servers
4. **Security Enforcement:** Allowlist validation
5. **Health Monitoring:** Track backend server status

### 2.2 Future Claude Desktop Configuration

**Simplified Configuration:**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.gateway.server"],
      "env": {
        "PROJECT_ROOT": "/path/to/Project_Sanctuary",
        "PYTHONPATH": "/path/to/Project_Sanctuary",
        "GATEWAY_CONFIG": "/path/to/Project_Sanctuary/.agent/config/gateway_config.json"
      },
      "cwd": "/path/to/Project_Sanctuary"
    }
  }
}
```

**Change:** 12 server entries → 1 gateway entry

### 2.3 Gateway Implementation

**New Directory Structure:**
```
Project_Sanctuary/
├── mcp_servers/
│   ├── gateway/
│   │   ├── server.py          # Gateway FastMCP server
│   │   ├── router.py          # Tool → Server routing
│   │   ├── registry.py        # Server registry (SQLite)
│   │   ├── proxy.py           # HTTP/stdio proxy client
│   │   ├── security.py        # Allowlist enforcement
│   │   └── config/
│   │       └── registry.db    # SQLite database
│   ├── adr/                   # Unchanged
│   ├── chronicle/             # Unchanged
│   ├── rag_cortex/            # Unchanged
│   └── ... (all backend servers unchanged)
```

**Gateway Server (Conceptual):**
```python
# mcp_servers/gateway/server.py
from fastmcp import FastMCP
from .router import Router
from .registry import ServerRegistry
from .proxy import ProxyClient
from .security import AllowlistEnforcer

mcp = FastMCP("project_sanctuary.gateway")

# Initialize components
registry = ServerRegistry("mcp_servers/gateway/config/registry.db")
router = Router(registry)
proxy = ProxyClient()
security = AllowlistEnforcer()

# Dynamic tool registration
# Gateway exposes ALL backend tools as its own tools
@mcp.tool()
async def cortex_query(query: str, max_results: int = 5, **kwargs) -> str:
    """Perform semantic search query against the knowledge base."""
    # 1. Security check
    if not security.validate("cortex_query", locals()):
        return json.dumps({"error": "Unauthorized"})
    
    # 2. Route to backend
    server = router.route("cortex_query")
    
    # 3. Proxy request
    response = await proxy.call(server, "cortex_query", {
        "query": query,
        "max_results": max_results,
        **kwargs
    })
    
    return response

# ... 62 more tools (one for each backend tool)

if __name__ == "__main__":
    mcp.run()
```

**Challenge:** Manually registering 63 tools is tedious. **Solution:** Dynamic tool generation.

### 2.4 Dynamic Tool Generation

**Better Approach:**
```python
# mcp_servers/gateway/server.py
from fastmcp import FastMCP

mcp = FastMCP("project_sanctuary.gateway")

# Load tool definitions from registry
tool_definitions = registry.get_all_tools()

# Dynamically create tools
for tool_def in tool_definitions:
    # Create wrapper function
    async def tool_wrapper(**kwargs):
        server = router.route(tool_def.name)
        return await proxy.call(server, tool_def.name, kwargs)
    
    # Set function metadata
    tool_wrapper.__name__ = tool_def.name
    tool_wrapper.__doc__ = tool_def.description
    
    # Register with FastMCP
    mcp.tool()(tool_wrapper)
```

**Problem:** FastMCP might not support dynamic registration. **Alternative:** Use FastMCP's internal API or generate code.

### 2.5 Registry Schema

**SQLite Database:**
```sql
-- Server registry
CREATE TABLE mcp_servers (
    name TEXT PRIMARY KEY,
    display_name TEXT,
    endpoint TEXT,           -- e.g., "http://localhost:8001" or "stdio"
    transport TEXT,          -- "http" or "stdio"
    status TEXT,             -- "running", "stopped", "error"
    last_health_check TIMESTAMP
);

-- Tool registry
CREATE TABLE mcp_tools (
    tool_name TEXT PRIMARY KEY,
    server_name TEXT,
    description TEXT,
    parameters JSON,
    FOREIGN KEY (server_name) REFERENCES mcp_servers(name)
);

-- Allowlist
CREATE TABLE allowlist (
    tool_name TEXT PRIMARY KEY,
    allowed BOOLEAN,
    require_approval BOOLEAN,
    max_rate_per_minute INTEGER
);
```

**Population:**
```sql
-- Example data
INSERT INTO mcp_servers VALUES
    ('rag_cortex', 'RAG Cortex MCP', 'http://localhost:8001', 'http', 'running', '2025-12-15 07:00:00'),
    ('git_workflow', 'Git Workflow MCP', 'stdio', 'stdio', 'running', '2025-12-15 07:00:00'),
    ('task', 'Task MCP', 'stdio', 'stdio', 'running', '2025-12-15 07:00:00');

INSERT INTO mcp_tools VALUES
    ('cortex_query', 'rag_cortex', 'Perform semantic search query', '{"query": "string", "max_results": "int"}'),
    ('git_add', 'git_workflow', 'Stage files for commit', '{"files": "list[str]"}'),
    ('create_task', 'task', 'Create a new task', '{"title": "string", "objective": "string", ...}');

INSERT INTO allowlist VALUES
    ('cortex_query', TRUE, FALSE, 100),
    ('git_add', TRUE, FALSE, 50),
    ('git_smart_commit', TRUE, TRUE, 10);  -- Requires approval
```

---

## 3. Migration Path

### 3.1 Phase 1: Gateway MVP (Week 1)

**Scope:** Basic gateway with 3 backend servers

**Steps:**
1. Create `mcp_servers/gateway/` directory
2. Implement `server.py` with FastMCP
3. Implement `registry.py` with SQLite
4. Implement `router.py` with static routing
5. Implement `proxy.py` with stdio support
6. Register 3 servers: rag_cortex, task, git_workflow
7. Test with Claude Desktop

**Deliverables:**
- Working gateway for 3 servers
- SQLite registry populated
- Basic routing logic
- stdio proxy working

**Success Criteria:**
- Can call `cortex_query` through gateway
- Can call `create_task` through gateway
- Can call `git_add` through gateway

### 3.2 Phase 2: Full Migration (Week 2-3)

**Scope:** All 12 servers migrated

**Steps:**
1. Add remaining 9 servers to registry
2. Implement HTTP proxy for remote servers (future)
3. Add security allowlist enforcement
4. Add health checks for all servers
5. Add circuit breaker pattern
6. Update Claude Desktop config (12 → 1 entry)
7. Full integration testing

**Deliverables:**
- All 63 tools accessible through gateway
- Security allowlist enforced
- Health monitoring active
- Updated documentation

**Success Criteria:**
- All existing workflows work through gateway
- Latency overhead <50ms
- No functionality regressions

### 3.3 Phase 3: Advanced Features (Week 4+)

**Scope:** Optimization and advanced patterns

**Steps:**
1. Implement Tool Search Tool (on-demand discovery)
2. Add caching layer for frequently used tools
3. Add metrics and monitoring (Prometheus)
4. Implement deferred loading for capability summaries
5. Add HTTP transport for remote servers
6. Performance optimization

**Deliverables:**
- Tool Search Tool working
- Caching reduces latency by 50%+
- Metrics dashboard available
- Remote server support ready

---

## 4. Compatibility Analysis

### 4.1 Backend Server Changes

**Required Changes:** **NONE**

**Rationale:**
- Backend servers remain unchanged
- Gateway acts as transparent proxy
- Existing FastMCP servers work as-is
- Can run in parallel during migration

**Validation:**
```bash
# Test backend server directly (bypasses gateway)
python -m mcp_servers.rag_cortex.server

# Test through gateway
python -m mcp_servers.gateway.server
```

### 4.2 Client Changes

**Required Changes:** Update Claude Desktop config

**Before:**
```json
{
  "mcpServers": {
    "rag_cortex": { ... },
    "git_workflow": { ... },
    "task": { ... },
    // ... 9 more
  }
}
```

**After:**
```json
{
  "mcpServers": {
    "sanctuary-broker": { ... }
  }
}
```

**Migration Strategy:**
1. Keep both configs during testing
2. Switch between them by renaming files
3. Validate all workflows work with gateway
4. Remove old config once confident

### 4.3 Testing Strategy

**Unit Tests:**
- Test router logic (tool → server mapping)
- Test proxy client (HTTP and stdio)
- Test security allowlist enforcement
- Test registry operations (CRUD)

**Integration Tests:**
- Test gateway with 1 backend server
- Test gateway with all 12 backend servers
- Test error handling (server down, timeout)
- Test concurrent requests

**E2E Tests:**
- Test real workflows through gateway
- Test with Claude Desktop
- Test latency and performance
- Test failure scenarios

---

## 5. Risk Analysis

### 5.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Gateway becomes bottleneck | High | Low | Connection pooling, async I/O |
| Latency overhead unacceptable | Medium | Low | Benchmark early, optimize hot paths |
| Dynamic tool registration fails | High | Medium | Fallback to static registration |
| Backend server compatibility | High | Low | No changes to backends required |
| SQLite registry corruption | Medium | Low | Regular backups, transaction safety |

### 5.2 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Gateway crashes | Critical | Low | Auto-restart, health checks |
| Configuration errors | High | Medium | Schema validation, dry-run mode |
| Migration breaks workflows | Critical | Medium | Parallel testing, rollback plan |
| User confusion | Low | High | Clear documentation, gradual rollout |

### 5.3 Rollback Plan

**If Gateway Fails:**
1. Revert Claude Desktop config to old version
2. Restart Claude Desktop
3. All backend servers still work independently
4. No data loss (registry is separate)

**Recovery Time:** <5 minutes

---

## 6. Benefits Analysis

### 6.1 Context Efficiency

**Before:**
- 12 servers × ~700 tokens/server = **8,400 tokens**
- 10% of 200K context window consumed upfront

**After:**
- 1 gateway × ~1,000 tokens = **1,000 tokens**
- 0.5% of context window consumed

**Savings:** 88% reduction in context overhead

### 6.2 Scalability

**Before:**
- Hard limit: ~20 servers before context saturation
- Adding server requires Claude Desktop config update

**After:**
- Soft limit: 100+ servers (limited by registry, not context)
- Adding server: Update registry only, no client changes

**Improvement:** 5x scalability increase

### 6.3 Flexibility

**Before:**
- Static configuration
- All servers loaded upfront
- No dynamic discovery

**After:**
- Dynamic configuration
- Servers loaded on-demand (future)
- Service discovery enabled

**Improvement:** Enables future cloud deployment, remote servers

### 6.4 Security

**Before:**
- No centralized security enforcement
- Each server handles own security
- Inconsistent patterns

**After:**
- Centralized allowlist enforcement
- Consistent security model
- Audit logging at gateway level

**Improvement:** Defense-in-depth, better visibility

---

## 7. Recommendations

### 7.1 Immediate Actions

1. ✅ **Create Task 110:** Track this initiative (already done)
2. ✅ **Research Phase:** Validate pattern viability (this document)
3. **Create ADR:** Formalize architectural decision
4. **Create Protocol 122:** Define Dynamic Binding Standard
5. **Build MVP:** Gateway with 3 servers

### 7.2 Architecture Decisions

**Recommendation 1: Use FastMCP for Gateway**
- **Rationale:** Consistent with existing servers, proven framework
- **Alternative:** Build custom MCP server (more work, no benefit)

**Recommendation 2: SQLite for Registry**
- **Rationale:** Lightweight, file-based, no external dependencies
- **Alternative:** etcd/Consul (overkill for 12 servers)

**Recommendation 3: Hybrid stdio/HTTP Transport**
- **Rationale:** stdio for Claude, HTTP for backends (future-proof)
- **Alternative:** stdio-to-stdio (simpler but less flexible)

**Recommendation 4: Static Tool Registration (Phase 1)**
- **Rationale:** Easier to implement, validate pattern first
- **Alternative:** Dynamic registration (more complex, save for Phase 2)

### 7.3 Success Metrics

**Phase 1 (MVP):**
- [ ] Gateway routes to 3 backend servers
- [ ] Latency overhead <30ms
- [ ] All tools work correctly
- [ ] No regressions in functionality

**Phase 2 (Full Migration):**
- [ ] All 12 servers migrated
- [ ] Context overhead reduced by 85%+
- [ ] Security allowlist enforced
- [ ] Health monitoring active

**Phase 3 (Advanced):**
- [ ] Tool Search Tool implemented
- [ ] Caching reduces latency by 50%+
- [ ] Metrics dashboard available
- [ ] Remote server support ready

---

## 8. Conclusion

**Current State:** Well-architected, consistent, production-ready MCP ecosystem with 12 independent servers.

**Future State:** Centralized gateway pattern that maintains all benefits while adding scalability, efficiency, and flexibility.

**Migration Risk:** **LOW** - Backend servers unchanged, client changes minimal, rollback trivial.

**Recommendation:** **PROCEED** with Gateway implementation.

**Next Steps:**
1. Create ADR 056 (already done)
2. Create Protocol 122: Dynamic Server Binding
3. Create Architecture Spec: Gateway Implementation
4. Build Phase 1 MVP (3 servers)
5. Validate with real workflows
6. Proceed to full migration

---

## 9. References

### Internal Documents
- `docs/architecture/mcp/architecture.md` - Current MCP architecture
- `mcp_servers/*/server.py` - Server implementations
- `claude_desktop_config.json` - Current configuration

### Research Documents
- `00_executive_summary.md` - Overall research findings
- `01_mcp_protocol_transport_layer.md` - Transport analysis
- `02_gateway_patterns_and_implementations.md` - Pattern research
- `03_performance_and_latency_analysis.md` - Performance benchmarks
- `04_security_architecture_and_threat_modeling.md` - Security analysis

### Related Sanctuary Documents
- Protocol 116: Ollama Container Network
- Task 087: MCP Operations Testing
- Chronicle 308: Doctrine of Successor State
