# Implementation Plan: MCP Gateway Architecture

**Research Date:** 2025-12-15  
**Task:** 110 - Dynamic MCP Gateway Architecture  
**Focus:** Detailed implementation roadmap, transition strategy, deliverables

---

## Executive Summary

This document provides a **comprehensive implementation plan** for migrating from the current static MCP server architecture to the Dynamic MCP Gateway pattern.

**Timeline:** 4 weeks (with IBM ContextForge) or 6-8 weeks (custom build)  
**Phases:** 5 phases (Planning → MVP → Migration → Optimization → Production)  
**Risk Level:** LOW (side-by-side deployment, easy rollback)  
**Team Size:** 1-2 developers

> **⚠️ IMPORTANT UPDATE (2025-12-15):**  
> After completing this implementation plan, we conducted a comprehensive "Build vs Buy vs Reuse" analysis (see `11_build_vs_buy_vs_reuse_analysis.md`) and decided to **reuse IBM ContextForge** instead of building from scratch.
> 
> **Decision:** ADR 057 - Adoption of IBM ContextForge for Dynamic MCP Gateway  
> **Rationale:** 2-3 weeks faster, $8K-16K cost savings, production-ready, 90% feature overlap
> 
> This document now serves **dual purposes:**
> 1. **Primary:** Reference architecture for ContextForge customization
> 2. **Fallback:** Complete custom build plan if ContextForge evaluation fails (Week 1 gate)

---

## 0. IBM ContextForge Integration Strategy

### 0.1 ContextForge Overview

**Repository:** https://github.com/IBM/mcp-context-forge  
**License:** Apache 2.0 (permissive, no vendor lock-in)  
**Version:** v0.9.0 (Nov 2025 - very active)  
**Adoption:** 3,000 stars, 434 forks, 88 contributors

**What ContextForge Provides:**
- ✅ Production-ready MCP Gateway with 400+ tests
- ✅ Service registry and dynamic routing
- ✅ Multi-transport support (stdio, HTTP, SSE, WebSocket)
- ✅ Built-in auth, retries, rate-limiting
- ✅ OpenTelemetry observability
- ✅ Admin UI for management
- ✅ Redis-backed caching and federation
- ✅ Docker/Kubernetes deployment ready

### 0.2 Implementation Approach: Fork & Customize

**Week 1: Fork, Deploy, Evaluate**
```bash
# 1. Fork the repository
git clone https://github.com/IBM/mcp-context-forge.git sanctuary-gateway
cd sanctuary-gateway

# 2. Install dependencies
pip install -e .

# 3. Configure for Sanctuary
cp .env.example .env
# Edit .env with Sanctuary-specific settings

# 4. Deploy MVP with 3 servers
# Configure rag_cortex, task, git_workflow as backend servers

# 5. Test with Claude Desktop
# Update claude_desktop_config.json to use ContextForge gateway

# 6. EVALUATION GATE (End of Week 1)
# - Feature gaps <50%? ✅ Continue with ContextForge
# - Feature gaps >50%? ❌ Pivot to custom build (Section 1-5)
# - Performance <50ms? ✅ Continue
# - Performance >50ms? ❌ Optimize or pivot
```

**Week 2-3: Customize for Sanctuary**
```python
# sanctuary-gateway/plugins/sanctuary_allowlist.py
"""
Custom plugin for Sanctuary-specific tool-level allowlist.
Integrates with Protocol 101 enforcement.
"""

class SanctuaryAllowlistPlugin:
    def __init__(self, config_path: str):
        self.allowlist = self.load_allowlist(config_path)
    
    def load_allowlist(self, path: str) -> dict:
        """Load project_mcp.json allowlist."""
        with open(path) as f:
            return json.load(f)
    
    async def validate_tool_call(self, tool_name: str, args: dict) -> bool:
        """Validate tool call against allowlist."""
        # Check if tool is allowed
        if tool_name not in self.allowlist.get("allowed_tools", []):
            raise PermissionError(f"Tool {tool_name} not in allowlist")
        
        # Check if operation requires approval
        if self.allowlist.get("operations", {}).get(tool_name, {}).get("approval_required"):
            # Integrate with Protocol 101 approval workflow
            return await self.request_approval(tool_name, args)
        
        return True
```

**Customization Areas:**
1. **Allowlist Plugin** - Tool-level security (Protocol 101)
2. **Registry Integration** - SQLite or use ContextForge's built-in
3. **Protocol 114 Integration** - Guardian Wakeup hooks
4. **Sanctuary-Specific Metadata** - Add Chronicle/ADR/Protocol tracking
5. **Custom Routing Logic** - If needed for special cases

**Week 4: Production Hardening**
- Migrate all 12 servers
- Full integration testing
- Performance optimization
- Monitoring setup (OpenTelemetry)
- Documentation updates

### 0.3 Fallback to Custom Build

**If ContextForge Doesn't Meet Needs:**
- Proceed with **Section 1-5** of this document (custom build)
- Timeline: 6-8 weeks instead of 4 weeks
- Cost: $24K-32K instead of $16K
- Risk: Higher (unproven architecture)

**Decision Point:** End of Week 1 evaluation

---

## 1. Implementation Phases (Custom Build Fallback)

### Phase 1: Planning & Documentation (Week 1)

**Objective:** Complete all planning artifacts before writing code

**Deliverables:**
1. ✅ **Research Complete** (6 documents in `research/RESEARCH_SUMMARIES/MCP_GATEWAY/`)
2. ✅ **ADR 056:** Adoption of Dynamic MCP Gateway Pattern
3. **Protocol 122:** Dynamic Server Binding Standard
4. **Architecture Specification:** `docs/mcp_gateway/ARCHITECTURE.md`
5. **Operations Reference:** `docs/mcp_gateway/OPERATIONS.md`
6. **Deployment Guide:** `docs/mcp_gateway/DEPLOYMENT.md`
7. **Testing Strategy:** `docs/mcp_gateway/TESTING.md`

**Success Criteria:**
- [ ] All documentation reviewed and approved
- [ ] Architecture diagrams created (Mermaid)
- [ ] Registry schema defined
- [ ] Security allowlist format defined

---

### Phase 2: Gateway MVP (Week 2)

**Objective:** Build minimal viable gateway with 3 backend servers

**Scope:**
- **Backend Servers:** rag_cortex, task, git_workflow (most used)
- **Transport:** stdio only (simplest)
- **Routing:** Static mapping (no registry yet)
- **Security:** Basic validation (no allowlist yet)

**Directory Structure:**
```
mcp_servers/
├── gateway/
│   ├── __init__.py
│   ├── server.py              # FastMCP gateway server
│   ├── router.py              # Static tool → server routing
│   ├── proxy.py               # stdio proxy client
│   ├── models.py              # Data models
│   └── config/
│       └── static_routes.json # Static routing config
└── ... (existing servers unchanged)
```

**Implementation Steps:**

**Step 1: Create Gateway Server (2 hours)**
```python
# mcp_servers/gateway/server.py
from fastmcp import FastMCP
from .router import StaticRouter
from .proxy import StdioProxy

mcp = FastMCP("project_sanctuary.gateway")
router = StaticRouter()
proxy = StdioProxy()

# Example: Proxy cortex_query
@mcp.tool()
async def cortex_query(query: str, max_results: int = 5, **kwargs) -> str:
    """Perform semantic search query against the knowledge base."""
    server = router.route("cortex_query")  # Returns "rag_cortex"
    response = await proxy.call(server, "cortex_query", {
        "query": query,
        "max_results": max_results,
        **kwargs
    })
    return response

# ... repeat for all tools in 3 servers (~25 tools)

if __name__ == "__main__":
    mcp.run()
```

**Step 2: Create Static Router (1 hour)**
```python
# mcp_servers/gateway/router.py
import json
from pathlib import Path

class StaticRouter:
    def __init__(self):
        config_path = Path(__file__).parent / "config" / "static_routes.json"
        with open(config_path) as f:
            self.routes = json.load(f)
    
    def route(self, tool_name: str) -> str:
        """Return server name for given tool."""
        if tool_name not in self.routes:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.routes[tool_name]
```

**Step 3: Create stdio Proxy (2 hours)**
```python
# mcp_servers/gateway/proxy.py
import subprocess
import json
import asyncio

class StdioProxy:
    def __init__(self):
        self.processes = {}  # server_name → subprocess
    
    async def call(self, server_name: str, tool_name: str, params: dict) -> str:
        """Call backend server via stdio."""
        # 1. Start server if not running
        if server_name not in self.processes:
            await self._start_server(server_name)
        
        # 2. Send JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": f"tools/{tool_name}",
            "params": params
        }
        
        proc = self.processes[server_name]
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        
        # 3. Read response
        response_line = proc.stdout.readline()
        response = json.loads(response_line)
        
        return response.get("result", "")
    
    async def _start_server(self, server_name: str):
        """Start backend MCP server as subprocess."""
        cmd = [
            "/path/to/.venv/bin/python",
            "-m",
            f"mcp_servers.{server_name}.server"
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self.processes[server_name] = proc
```

**Step 4: Create Static Routes Config (30 minutes)**
```json
{
  "cortex_query": "rag_cortex",
  "cortex_ingest_full": "rag_cortex",
  "cortex_get_stats": "rag_cortex",
  "create_task": "task",
  "update_task": "task",
  "get_task": "task",
  "git_add": "git_workflow",
  "git_smart_commit": "git_workflow",
  "git_get_status": "git_workflow"
}
```

**Step 5: Test with Claude Desktop (2 hours)**
1. Update `claude_desktop_config.json` (add gateway entry)
2. Restart Claude Desktop
3. Test each tool through gateway
4. Verify responses match direct calls

**Success Criteria:**
- [ ] Gateway routes to 3 backend servers
- [ ] All 25 tools work correctly
- [ ] Latency overhead <50ms
- [ ] No functionality regressions

**Estimated Time:** 8-10 hours (1-2 days)

---

### Phase 3: Registry Implementation (Week 3)

**Objective:** Replace static routing with SQLite registry

**Deliverables:**
1. SQLite database schema
2. Registry CRUD operations
3. Dynamic tool registration
4. Health check system

**Directory Structure:**
```
mcp_servers/gateway/
├── server.py              # Updated to use registry
├── router.py              # Updated to query registry
├── registry.py            # NEW: Registry operations
├── health.py              # NEW: Health checks
└── config/
    ├── registry.db        # NEW: SQLite database
    └── schema.sql         # NEW: Database schema
```

**Implementation Steps:**

**Step 1: Define Registry Schema (1 hour)**
```sql
-- mcp_servers/gateway/config/schema.sql

-- Server registry
CREATE TABLE mcp_servers (
    name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    module_path TEXT NOT NULL,        -- e.g., "mcp_servers.rag_cortex.server"
    transport TEXT NOT NULL,           -- "stdio" or "http"
    endpoint TEXT,                     -- NULL for stdio, URL for http
    status TEXT DEFAULT 'stopped',     -- "running", "stopped", "error"
    last_health_check TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tool registry
CREATE TABLE mcp_tools (
    tool_name TEXT PRIMARY KEY,
    server_name TEXT NOT NULL,
    description TEXT,
    parameters_schema JSON,            -- JSON schema for parameters
    is_read_only BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (server_name) REFERENCES mcp_servers(name)
);

-- Allowlist
CREATE TABLE allowlist (
    tool_name TEXT PRIMARY KEY,
    allowed BOOLEAN DEFAULT 1,
    require_approval BOOLEAN DEFAULT 0,
    max_rate_per_minute INTEGER DEFAULT 100,
    notes TEXT,
    FOREIGN KEY (tool_name) REFERENCES mcp_tools(tool_name)
);

-- Audit log
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tool_name TEXT NOT NULL,
    server_name TEXT NOT NULL,
    user TEXT DEFAULT 'claude_desktop',
    params JSON,
    latency_ms INTEGER,
    status TEXT,                       -- "success", "error"
    error_message TEXT
);

-- Indexes
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_tool ON audit_log(tool_name);
CREATE INDEX idx_tools_server ON mcp_tools(server_name);
```

**Step 2: Populate Registry (1 hour)**
```sql
-- Insert servers
INSERT INTO mcp_servers (name, display_name, module_path, transport) VALUES
    ('rag_cortex', 'RAG Cortex MCP', 'mcp_servers.rag_cortex.server', 'stdio'),
    ('task', 'Task MCP', 'mcp_servers.task.server', 'stdio'),
    ('git_workflow', 'Git Workflow MCP', 'mcp_servers.git.server', 'stdio');

-- Insert tools (rag_cortex)
INSERT INTO mcp_tools (tool_name, server_name, description, is_read_only) VALUES
    ('cortex_query', 'rag_cortex', 'Perform semantic search query', 1),
    ('cortex_ingest_full', 'rag_cortex', 'Full re-ingestion of knowledge base', 0),
    ('cortex_get_stats', 'rag_cortex', 'Get database statistics', 1),
    ('cortex_ingest_incremental', 'rag_cortex', 'Incrementally ingest documents', 0),
    ('cortex_cache_get', 'rag_cortex', 'Retrieve cached answer', 1),
    ('cortex_cache_set', 'rag_cortex', 'Store answer in cache', 0),
    ('cortex_cache_warmup', 'rag_cortex', 'Pre-populate cache', 0),
    ('cortex_guardian_wakeup', 'rag_cortex', 'Generate Guardian boot digest', 0),
    ('cortex_cache_stats', 'rag_cortex', 'Get cache statistics', 1);

-- Insert tools (task)
INSERT INTO mcp_tools (tool_name, server_name, description, is_read_only) VALUES
    ('create_task', 'task', 'Create a new task', 0),
    ('update_task', 'task', 'Update existing task', 0),
    ('update_task_status', 'task', 'Change task status', 0),
    ('get_task', 'task', 'Retrieve specific task', 1),
    ('list_tasks', 'task', 'List tasks with filters', 1),
    ('search_tasks', 'task', 'Search tasks by content', 1);

-- Insert tools (git_workflow)
INSERT INTO mcp_tools (tool_name, server_name, description, is_read_only) VALUES
    ('git_smart_commit', 'git_workflow', 'Commit with P101 enforcement', 0),
    ('git_get_safety_rules', 'git_workflow', 'Get Git safety rules', 1),
    ('git_get_status', 'git_workflow', 'Get repository status', 1),
    ('git_add', 'git_workflow', 'Stage files for commit', 0),
    ('git_push_feature', 'git_workflow', 'Push feature branch', 0),
    ('git_start_feature', 'git_workflow', 'Start new feature branch', 0),
    ('git_finish_feature', 'git_workflow', 'Finish feature branch', 0),
    ('git_diff', 'git_workflow', 'Show changes', 1),
    ('git_log', 'git_workflow', 'Show commit history', 1);

-- Insert allowlist (default: all allowed)
INSERT INTO allowlist (tool_name, allowed, require_approval) 
SELECT tool_name, 1, 0 FROM mcp_tools;

-- Override: Require approval for destructive operations
UPDATE allowlist SET require_approval = 1 WHERE tool_name IN (
    'git_smart_commit',
    'git_push_feature',
    'cortex_ingest_full'
);
```

**Step 3: Implement Registry Operations (2 hours)**
```python
# mcp_servers/gateway/registry.py
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
import json

class ServerRegistry:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database from schema if not exists."""
        if not Path(self.db_path).exists():
            schema_path = Path(__file__).parent / "config" / "schema.sql"
            with open(schema_path) as f:
                schema = f.read()
            
            conn = sqlite3.connect(self.db_path)
            conn.executescript(schema)
            conn.close()
    
    def route(self, tool_name: str) -> str:
        """Get server name for given tool."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT server_name FROM mcp_tools WHERE tool_name = ?",
            (tool_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return result[0]
    
    def get_server(self, server_name: str) -> Optional[Dict]:
        """Get server configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM mcp_servers WHERE name = ?",
            (server_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    def get_all_tools(self) -> List[Dict]:
        """Get all registered tools."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM mcp_tools")
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def is_allowed(self, tool_name: str) -> bool:
        """Check if tool is in allowlist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT allowed FROM allowlist WHERE tool_name = ?",
            (tool_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else False
    
    def requires_approval(self, tool_name: str) -> bool:
        """Check if tool requires human approval."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT require_approval FROM allowlist WHERE tool_name = ?",
            (tool_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else False
    
    def log_call(self, tool_name: str, server_name: str, latency_ms: int, status: str, error: str = None):
        """Log tool call to audit log."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO audit_log (tool_name, server_name, latency_ms, status, error_message)
               VALUES (?, ?, ?, ?, ?)""",
            (tool_name, server_name, latency_ms, status, error)
        )
        conn.commit()
        conn.close()
```

**Step 4: Update Router to Use Registry (30 minutes)**
```python
# mcp_servers/gateway/router.py (updated)
from .registry import ServerRegistry

class Router:
    def __init__(self, registry: ServerRegistry):
        self.registry = registry
    
    def route(self, tool_name: str) -> str:
        """Route tool to appropriate server using registry."""
        return self.registry.route(tool_name)
```

**Step 5: Add Health Checks (2 hours)**
```python
# mcp_servers/gateway/health.py
import asyncio
from datetime import datetime
from .registry import ServerRegistry
from .proxy import StdioProxy

class HealthChecker:
    def __init__(self, registry: ServerRegistry, proxy: StdioProxy):
        self.registry = registry
        self.proxy = proxy
    
    async def check_server(self, server_name: str) -> bool:
        """Check if server is healthy."""
        try:
            # Try to call a read-only operation
            # (Assumes all servers have a health check endpoint)
            await asyncio.wait_for(
                self.proxy.call(server_name, "health_check", {}),
                timeout=5.0
            )
            return True
        except Exception:
            return False
    
    async def check_all_servers(self):
        """Check health of all servers and update registry."""
        servers = self.registry.get_all_servers()
        
        for server in servers:
            is_healthy = await self.check_server(server['name'])
            status = "running" if is_healthy else "error"
            
            self.registry.update_server_status(
                server['name'],
                status,
                datetime.now()
            )
```

**Success Criteria:**
- [ ] Registry database created and populated
- [ ] Router uses registry for all lookups
- [ ] Health checks run every 60 seconds
- [ ] Audit log captures all tool calls

**Estimated Time:** 8-10 hours (1-2 days)

---

### Phase 4: Full Migration (Week 4)

**Objective:** Migrate all 12 servers to Gateway

**Scope:**
- Add remaining 9 servers to registry
- Register all 63 tools
- Update allowlist for all tools
- Full integration testing

**Implementation Steps:**

**Step 1: Add Remaining Servers to Registry (2 hours)**
```sql
INSERT INTO mcp_servers (name, display_name, module_path, transport) VALUES
    ('adr', 'ADR MCP', 'mcp_servers.adr.server', 'stdio'),
    ('chronicle', 'Chronicle MCP', 'mcp_servers.chronicle.server', 'stdio'),
    ('protocol', 'Protocol MCP', 'mcp_servers.protocol.server', 'stdio'),
    ('council', 'Council MCP', 'mcp_servers.council.server', 'stdio'),
    ('agent_persona', 'Agent Persona MCP', 'mcp_servers.agent_persona.server', 'stdio'),
    ('forge_llm', 'Forge LLM MCP', 'mcp_servers.forge_llm.server', 'stdio'),
    ('config', 'Config MCP', 'mcp_servers.config.server', 'stdio'),
    ('code', 'Code MCP', 'mcp_servers.code.server', 'stdio'),
    ('orchestrator', 'Orchestrator MCP', 'mcp_servers.orchestrator.server', 'stdio');

-- Insert all tools (38 more tools)
-- ... (see full SQL in appendix)
```

**Step 2: Update Gateway Server with All Tools (4 hours)**

**Option A: Manual Registration (tedious but explicit)**
```python
# Manually register all 63 tools
@mcp.tool()
async def adr_create(...): ...

@mcp.tool()
async def adr_get(...): ...

# ... 61 more
```

**Option B: Dynamic Registration (recommended)**
```python
# mcp_servers/gateway/server.py (dynamic version)
from fastmcp import FastMCP
from .registry import ServerRegistry
from .router import Router
from .proxy import StdioProxy
import inspect

mcp = FastMCP("project_sanctuary.gateway")
registry = ServerRegistry("mcp_servers/gateway/config/registry.db")
router = Router(registry)
proxy = StdioProxy()

# Get all tools from registry
all_tools = registry.get_all_tools()

# Dynamically create wrapper functions
for tool in all_tools:
    tool_name = tool['tool_name']
    description = tool['description']
    
    # Create wrapper function
    async def tool_wrapper(**kwargs):
        server = router.route(tool_name)
        return await proxy.call(server, tool_name, kwargs)
    
    # Set metadata
    tool_wrapper.__name__ = tool_name
    tool_wrapper.__doc__ = description
    
    # Register with FastMCP
    mcp.tool()(tool_wrapper)

if __name__ == "__main__":
    mcp.run()
```

**Step 3: Side-by-Side Deployment (2 hours)**

**Create Dual Configuration:**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "Sanctuary Gateway (NEW)",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.gateway.server"],
      "env": {
        "PROJECT_ROOT": "/path/to/Project_Sanctuary",
        "PYTHONPATH": "/path/to/Project_Sanctuary"
      },
      "cwd": "/path/to/Project_Sanctuary"
    },
    "rag_cortex_legacy": {
      "displayName": "RAG Cortex MCP (LEGACY)",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.rag_cortex.server"],
      "env": { ... },
      "cwd": "/path/to/Project_Sanctuary"
    }
    // ... other legacy servers
  }
}
```

**Benefits:**
- Can compare gateway vs direct calls
- Easy rollback if issues found
- Gradual migration possible

**Step 4: Integration Testing (4 hours)**

**Test Matrix:**
| Server | Tools | Test Status |
|--------|-------|-------------|
| rag_cortex | 9 | ✅ |
| task | 6 | ✅ |
| git_workflow | 9 | ✅ |
| adr | 5 | ⏳ |
| chronicle | 6 | ⏳ |
| protocol | 5 | ⏳ |
| council | 2 | ⏳ |
| agent_persona | 5 | ⏳ |
| forge_llm | 2 | ⏳ |
| config | 4 | ⏳ |
| code | 9 | ⏳ |
| orchestrator | 2 | ⏳ |

**Test Script:**
```python
# tests/mcp_servers/gateway/test_full_migration.py
import pytest
from mcp_servers.gateway.server import mcp
from mcp_servers.gateway.registry import ServerRegistry

@pytest.fixture
def registry():
    return ServerRegistry("mcp_servers/gateway/config/registry.db")

def test_all_tools_registered(registry):
    """Verify all 63 tools are in registry."""
    tools = registry.get_all_tools()
    assert len(tools) == 63

def test_all_servers_registered(registry):
    """Verify all 12 servers are in registry."""
    servers = registry.get_all_servers()
    assert len(servers) == 12

@pytest.mark.asyncio
async def test_cortex_query_through_gateway():
    """Test cortex_query through gateway."""
    result = await mcp.call_tool("cortex_query", {
        "query": "What is Protocol 101?",
        "max_results": 3
    })
    assert "Protocol 101" in result

# ... 62 more tool tests
```

**Success Criteria:**
- [ ] All 12 servers registered
- [ ] All 63 tools accessible through gateway
- [ ] All integration tests pass
- [ ] Latency overhead <50ms
- [ ] No functionality regressions

**Estimated Time:** 12-16 hours (2-3 days)

---

### Phase 5: Production Hardening (Week 5-6)

**Objective:** Add production-ready features

**Deliverables:**
1. Security allowlist enforcement
2. Monitoring and metrics
3. Circuit breaker pattern
4. Performance optimization
5. Documentation updates

**Implementation Steps:**

**Step 1: Security Enforcement (4 hours)**
```python
# mcp_servers/gateway/security.py
from .registry import ServerRegistry

class SecurityEnforcer:
    def __init__(self, registry: ServerRegistry):
        self.registry = registry
    
    async def validate(self, tool_name: str, params: dict) -> bool:
        """Validate tool call against allowlist."""
        # 1. Check if tool is allowed
        if not self.registry.is_allowed(tool_name):
            raise SecurityError(f"Tool {tool_name} is not in allowlist")
        
        # 2. Check if approval required
        if self.registry.requires_approval(tool_name):
            approved = await self.request_approval(tool_name, params)
            if not approved:
                raise SecurityError(f"Approval denied for {tool_name}")
        
        return True
    
    async def request_approval(self, tool_name: str, params: dict) -> bool:
        """Request human approval for tool call."""
        print(f"\n⚠️  APPROVAL REQUIRED ⚠️")
        print(f"Tool: {tool_name}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        response = input("Approve? (yes/no): ")
        return response.lower() == "yes"
```

**Step 2: Monitoring (4 hours)**
```python
# mcp_servers/gateway/metrics.py
from prometheus_client import Histogram, Counter, Gauge
import time

# Metrics
request_latency = Histogram(
    'gateway_request_latency_seconds',
    'Gateway request latency',
    ['tool_name', 'server_name']
)

request_count = Counter(
    'gateway_requests_total',
    'Total gateway requests',
    ['tool_name', 'server_name', 'status']
)

active_requests = Gauge(
    'gateway_active_requests',
    'Number of active requests'
)

class MetricsCollector:
    async def track_call(self, tool_name: str, server_name: str, func):
        """Track metrics for tool call."""
        start = time.time()
        active_requests.inc()
        
        try:
            result = await func()
            request_count.labels(tool_name, server_name, 'success').inc()
            return result
        except Exception as e:
            request_count.labels(tool_name, server_name, 'error').inc()
            raise e
        finally:
            latency = time.time() - start
            request_latency.labels(tool_name, server_name).observe(latency)
            active_requests.dec()
```

**Step 3: Circuit Breaker (3 hours)**
```python
# mcp_servers/gateway/circuit_breaker.py
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = {}
        self.state = {}
        self.open_until = {}
    
    async def call(self, server_name: str, func):
        """Call function with circuit breaker protection."""
        # Check circuit state
        if self._is_open(server_name):
            raise Exception(f"Circuit breaker open for {server_name}")
        
        try:
            result = await func()
            self._on_success(server_name)
            return result
        except Exception as e:
            self._on_failure(server_name)
            raise e
    
    def _is_open(self, server_name: str) -> bool:
        if server_name not in self.open_until:
            return False
        
        if time.time() < self.open_until[server_name]:
            return True
        
        # Timeout expired, try half-open
        self.state[server_name] = CircuitState.HALF_OPEN
        return False
    
    def _on_success(self, server_name: str):
        self.failure_count[server_name] = 0
        self.state[server_name] = CircuitState.CLOSED
    
    def _on_failure(self, server_name: str):
        count = self.failure_count.get(server_name, 0) + 1
        self.failure_count[server_name] = count
        
        if count >= self.failure_threshold:
            self.state[server_name] = CircuitState.OPEN
            self.open_until[server_name] = time.time() + self.timeout
```

**Step 4: Performance Optimization (4 hours)**

**Connection Pooling:**
```python
# mcp_servers/gateway/proxy.py (optimized)
import httpx

class OptimizedProxy:
    def __init__(self):
        # HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(10.0)
        )
        
        # stdio process pool
        self.stdio_processes = {}
    
    async def call(self, server_name: str, tool_name: str, params: dict) -> str:
        server = self.registry.get_server(server_name)
        
        if server['transport'] == 'http':
            return await self._call_http(server['endpoint'], tool_name, params)
        else:
            return await self._call_stdio(server_name, tool_name, params)
```

**Caching:**
```python
# mcp_servers/gateway/cache.py
from functools import lru_cache
import hashlib
import json
import time

class ResponseCache:
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, tool_name: str, params: dict):
        """Get cached response if available."""
        key = self._make_key(tool_name, params)
        
        if key in self.cache:
            cached, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return cached
        
        return None
    
    def set(self, tool_name: str, params: dict, response: str):
        """Cache response."""
        key = self._make_key(tool_name, params)
        self.cache[key] = (response, time.time())
    
    def _make_key(self, tool_name: str, params: dict) -> str:
        """Generate cache key."""
        data = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()
```

**Step 5: Documentation Updates (4 hours)**

**Update Existing Docs:**
- `docs/mcp/architecture.md` → Add Gateway section
- `docs/mcp/QUICKSTART.md` → Update with Gateway instructions
- `README.md` → Update architecture diagram

**Create New Docs:**
- `docs/mcp_gateway/ARCHITECTURE.md` → Gateway architecture
- `docs/mcp_gateway/OPERATIONS.md` → Operations reference
- `docs/mcp_gateway/DEPLOYMENT.md` → Deployment guide
- `docs/mcp_gateway/TESTING.md` → Testing guide
- `docs/mcp_gateway/TROUBLESHOOTING.md` → Common issues

**Success Criteria:**
- [ ] Security allowlist enforced
- [ ] Metrics dashboard available
- [ ] Circuit breaker prevents cascading failures
- [ ] Cache hit rate >50% for read operations
- [ ] All documentation updated

**Estimated Time:** 20-24 hours (3-4 days)

---

## 2. Deliverables Checklist

### Documentation Deliverables

**Research Phase (✅ Complete):**
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/00_executive_summary.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/01_mcp_protocol_transport_layer.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/02_gateway_patterns_and_implementations.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/03_performance_and_latency_analysis.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/04_security_architecture_and_threat_modeling.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/05_current_vs_future_state_architecture.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/06_benefits_analysis.md`
- [x] `research/RESEARCH_SUMMARIES/MCP_GATEWAY/07_implementation_plan.md` (this document)

**Formalization Phase (In Progress):**
- [x] `ADRs/056_adoption_of_dynamic_mcp_gateway_pattern.md`
- [ ] `01_PROTOCOLS/122_dynamic_server_binding.md`

**Architecture Phase (Pending):**
- [ ] `docs/mcp_gateway/ARCHITECTURE.md`
- [ ] `docs/mcp_gateway/OPERATIONS.md`
- [ ] `docs/mcp_gateway/DEPLOYMENT.md`
- [ ] `docs/mcp_gateway/TESTING.md`
- [ ] `docs/mcp_gateway/TROUBLESHOOTING.md`

### Code Deliverables

**Gateway Core:**
- [ ] `mcp_servers/gateway/__init__.py`
- [ ] `mcp_servers/gateway/server.py`
- [ ] `mcp_servers/gateway/router.py`
- [ ] `mcp_servers/gateway/registry.py`
- [ ] `mcp_servers/gateway/proxy.py`
- [ ] `mcp_servers/gateway/security.py`
- [ ] `mcp_servers/gateway/health.py`
- [ ] `mcp_servers/gateway/metrics.py`
- [ ] `mcp_servers/gateway/circuit_breaker.py`
- [ ] `mcp_servers/gateway/cache.py`
- [ ] `mcp_servers/gateway/models.py`

**Configuration:**
- [ ] `mcp_servers/gateway/config/schema.sql`
- [ ] `mcp_servers/gateway/config/registry.db`
- [ ] `mcp_servers/gateway/config/populate_registry.sql`

**Testing:**
- [ ] `tests/mcp_servers/gateway/test_router.py`
- [ ] `tests/mcp_servers/gateway/test_registry.py`
- [ ] `tests/mcp_servers/gateway/test_proxy.py`
- [ ] `tests/mcp_servers/gateway/test_security.py`
- [ ] `tests/mcp_servers/gateway/test_integration.py`
- [ ] `tests/mcp_servers/gateway/test_full_migration.py`

---

## 3. Testing Strategy

### 3.1 Unit Tests

**Coverage Target:** 80%+

**Test Files:**
```
tests/mcp_servers/gateway/
├── test_router.py              # Router logic
├── test_registry.py            # Registry CRUD
├── test_proxy.py               # stdio/HTTP proxy
├── test_security.py            # Allowlist enforcement
├── test_health.py              # Health checks
├── test_circuit_breaker.py     # Circuit breaker
└── test_cache.py               # Response caching
```

**Example Test:**
```python
# tests/mcp_servers/gateway/test_router.py
import pytest
from mcp_servers.gateway.router import Router
from mcp_servers.gateway.registry import ServerRegistry

@pytest.fixture
def registry():
    return ServerRegistry(":memory:")  # In-memory DB for tests

@pytest.fixture
def router(registry):
    return Router(registry)

def test_route_known_tool(router, registry):
    """Test routing for known tool."""
    # Setup
    registry.add_tool("cortex_query", "rag_cortex", "Query cortex")
    
    # Execute
    server = router.route("cortex_query")
    
    # Verify
    assert server == "rag_cortex"

def test_route_unknown_tool(router):
    """Test routing for unknown tool raises error."""
    with pytest.raises(ValueError, match="Unknown tool"):
        router.route("nonexistent_tool")
```

### 3.2 Integration Tests

**Test Scenarios:**
1. Gateway → Single Backend Server
2. Gateway → Multiple Backend Servers
3. Gateway → Backend Server (with failure)
4. Gateway → Backend Server (with retry)
5. Gateway → Backend Server (with caching)

**Example Test:**
```python
# tests/mcp_servers/gateway/test_integration.py
import pytest
from mcp_servers.gateway.server import mcp

@pytest.mark.asyncio
async def test_cortex_query_integration():
    """Test cortex_query through full gateway stack."""
    result = await mcp.call_tool("cortex_query", {
        "query": "What is Protocol 101?",
        "max_results": 3
    })
    
    # Verify response format
    assert isinstance(result, str)
    data = json.loads(result)
    assert "results" in data
    assert len(data["results"]) <= 3
```

### 3.3 E2E Tests

**Test with Claude Desktop:**
1. Start Gateway
2. Update Claude Desktop config
3. Restart Claude Desktop
4. Execute test prompts
5. Verify responses

**Test Prompts:**
```
1. "What is Protocol 101?" (tests cortex_query)
2. "Create a new task for testing the gateway" (tests create_task)
3. "What's the current git status?" (tests git_get_status)
4. "List all protocols" (tests protocol_list)
5. "Show recent chronicle entries" (tests chronicle_list_entries)
```

### 3.4 Performance Tests

**Latency Benchmarks:**
```python
# tests/mcp_servers/gateway/test_performance.py
import pytest
import time

@pytest.mark.asyncio
async def test_gateway_latency_overhead():
    """Measure latency overhead of gateway."""
    # Direct call (baseline)
    start = time.time()
    await direct_cortex_query("test")
    direct_latency = time.time() - start
    
    # Gateway call
    start = time.time()
    await gateway_cortex_query("test")
    gateway_latency = time.time() - start
    
    # Verify overhead <50ms
    overhead = gateway_latency - direct_latency
    assert overhead < 0.050  # 50ms
```

**Load Tests:**
```python
@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test gateway under concurrent load."""
    # Send 100 concurrent requests
    tasks = [
        mcp.call_tool("cortex_query", {"query": f"test {i}"})
        for i in range(100)
    ]
    
    start = time.time()
    results = await asyncio.gather(*tasks)
    duration = time.time() - start
    
    # Verify all succeeded
    assert len(results) == 100
    assert all(r for r in results)
    
    # Verify throughput >10 RPS
    throughput = 100 / duration
    assert throughput > 10
```

---

## 4. Deployment Architecture

> **Note on Container Runtimes:** While this document uses **Podman** in examples, the architecture is **container-runtime agnostic**. The same deployment patterns work with **Docker**, **Kubernetes**, and **OpenShift**. See Section 4.3 for detailed comparison and migration paths.

### 4.1 Future State Architecture Diagram

![mcp_gateway_legacy_migration_target](docs/architecture_diagrams/system/mcp_gateway_legacy_migration_target.png)

*[Source: mcp_gateway_legacy_migration_target.mmd](docs/architecture_diagrams/system/mcp_gateway_legacy_migration_target.mmd)*

### 4.2 Container Deployment

> **Container Runtime Flexibility:** The configuration below uses Podman Compose syntax, but is compatible with Docker Compose with minimal changes. For Kubernetes/OpenShift, see equivalent manifests in Section 4.3.

**Podman/Docker Compose Configuration:**
```yaml
# podman-compose.yml or docker-compose.yml
version: '3'

services:
  sanctuary-broker:
    build: ./mcp_servers/gateway
    container_name: sanctuary-broker-mcp
    ports:
      - "9000:9000"  # Management API
    volumes:
      - ./mcp_servers/gateway/config:/app/config
      - ./logs:/app/logs
    networks:
      - sanctuary-internal
    environment:
      - PROJECT_ROOT=/app
      - REGISTRY_DB=/app/config/registry.db
    restart: unless-stopped
  
  rag-cortex:
    build: ./mcp_servers/rag_cortex
    container_name: rag-cortex-mcp
    ports:
      - "8001:8001"
    networks:
      - sanctuary-internal
    environment:
      - PROJECT_ROOT=/app
      - CHROMA_HOST=chroma-mcp
      - CHROMA_PORT=8000
    depends_on:
      - chroma
      - ollama
  
  task:
    build: ./mcp_servers/task
    container_name: task-mcp
    ports:
      - "8002:8002"
    networks:
      - sanctuary-internal
    volumes:
      - ./TASKS:/app/TASKS
  
  # ... (10 more backend servers)
  
  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-mcp
    ports:
      - "8000:8000"
    networks:
      - sanctuary-internal
    volumes:
      - chroma-data:/chroma/chroma
  
  ollama:
    image: ollama/ollama:latest
    container_name: ollama_model_mcp
    ports:
      - "11434:11434"
    networks:
      - sanctuary-internal
    volumes:
      - ollama-data:/root/.ollama

networks:
  sanctuary-internal:
    driver: bridge

volumes:
  chroma-data:
  ollama-data:
```

### 4.3 Deployment Modes

**Mode 1: Development (stdio, no containers)**
```
Claude Desktop → Gateway (stdio) → Backend Servers (stdio)
```

**Mode 2: Hybrid (stdio + HTTP)**
```
Claude Desktop → Gateway (stdio) → Backend Servers (HTTP in containers)
```

**Mode 3: Production (HTTP + containers)**
```
Claude Desktop → Gateway (HTTP) → Backend Servers (HTTP in containers)
```

### 4.4 Container Runtime Comparison

#### **Podman (Recommended for Local/Single-Host)**

**Pros:**
- ✅ **Rootless containers** - Better security (no root daemon)
- ✅ **Daemonless** - Simpler architecture, no background process
- ✅ **Docker-compatible CLI** - Easy migration from Docker
- ✅ **Systemd integration** - Native service management
- ✅ **Pod support** - Kubernetes-like pod concept
- ✅ **No vendor lock-in** - Open-source, community-driven

**Cons:**
- ⚠️ **Smaller ecosystem** - Fewer third-party tools than Docker
- ⚠️ **Compose limitations** - Some Docker Compose features missing
- ⚠️ **Less mature** - Newer than Docker

**Use Cases:**
- Local development
- Single-host production deployments
- Security-conscious environments
- Red Hat/Fedora/CentOS ecosystems

**Example Deployment:**
```bash
# Install Podman
sudo dnf install podman podman-compose

# Start Gateway
podman-compose up -d

# Enable systemd service
podman generate systemd --name sanctuary-broker-mcp > /etc/systemd/system/sanctuary-broker.service
systemctl enable --now sanctuary-broker
```

---

#### **Docker (Alternative for Local/Single-Host)**

**Pros:**
- ✅ **Largest ecosystem** - Most third-party tools, integrations
- ✅ **Mature** - Battle-tested, stable
- ✅ **Docker Compose** - Rich feature set, widely used
- ✅ **Extensive documentation** - Large community, many tutorials
- ✅ **Docker Hub** - Massive image registry

**Cons:**
- ⚠️ **Requires daemon** - Background process with root privileges
- ⚠️ **Vendor lock-in** - Owned by Docker Inc.
- ⚠️ **Security concerns** - Root daemon is attack surface

**Use Cases:**
- Teams already using Docker
- Need for Docker-specific tools
- Maximum compatibility with third-party images

**Example Deployment:**
```bash
# Install Docker
sudo apt-get install docker.io docker-compose

# Start Gateway
docker-compose up -d

# View logs
docker-compose logs -f sanctuary-broker
```

**Migration from Podman:**
```bash
# Minimal changes needed
mv podman-compose.yml docker-compose.yml
# Change: podman → docker in commands
```

---

#### **Kubernetes (Recommended for Multi-Host/Cloud)**

**Pros:**
- ✅ **Production-grade orchestration** - Industry standard
- ✅ **Auto-scaling** - Horizontal pod autoscaling (HPA)
- ✅ **Self-healing** - Automatic restart, rescheduling
- ✅ **Multi-node** - Distributed deployments
- ✅ **Service discovery** - Built-in DNS, load balancing
- ✅ **Rolling updates** - Zero-downtime deployments
- ✅ **Cloud-native** - AWS EKS, GCP GKE, Azure AKS support

**Cons:**
- ⚠️ **Complexity** - Steep learning curve
- ⚠️ **Overkill for small scale** - Too much for 12 servers
- ⚠️ **Resource overhead** - Control plane requires resources
- ⚠️ **Operational burden** - Requires K8s expertise

**Use Cases:**
- Multi-host deployments
- Cloud environments (AWS, GCP, Azure)
- High availability requirements
- Auto-scaling needs
- Enterprise production

**Example Deployment:**
```yaml
# kubernetes/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sanctuary-broker
  namespace: sanctuary
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sanctuary-broker
  template:
    metadata:
      labels:
        app: sanctuary-broker
    spec:
      containers:
      - name: gateway
        image: sanctuary-broker-mcp:latest
        ports:
        - containerPort: 9000
        env:
        - name: PROJECT_ROOT
          value: "/app"
        - name: REGISTRY_DB
          value: "/app/config/registry.db"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: sanctuary-broker-config
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: sanctuary-broker
  namespace: sanctuary
spec:
  selector:
    app: sanctuary-broker
  ports:
  - protocol: TCP
    port: 9000
    targetPort: 9000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sanctuary-broker-hpa
  namespace: sanctuary
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sanctuary-broker
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Deploy to Kubernetes:**
```bash
# Create namespace
kubectl create namespace sanctuary

# Apply manifests
kubectl apply -f kubernetes/

# Check status
kubectl get pods -n sanctuary
kubectl get svc -n sanctuary

# View logs
kubectl logs -f deployment/sanctuary-broker -n sanctuary
```

---

#### **OpenShift (Enterprise Kubernetes)**

**Pros:**
- ✅ **Enterprise Kubernetes** - Red Hat-supported
- ✅ **Built-in security** - SELinux, RBAC, network policies
- ✅ **Integrated CI/CD** - Built-in pipelines, image registry
- ✅ **Developer-friendly** - Web console, CLI tools
- ✅ **Multi-tenancy** - Project isolation, quotas
- ✅ **Certified operators** - Pre-built application templates

**Cons:**
- ⚠️ **Commercial** - Requires Red Hat subscription
- ⚠️ **Complexity** - Even more complex than K8s
- ⚠️ **Overkill** - Too much for current scale
- ⚠️ **Cost** - Licensing fees

**Use Cases:**
- Enterprise environments
- Red Hat ecosystem
- Strict compliance requirements
- Need for commercial support

**Example Deployment:**
```yaml
# openshift/gateway-deploymentconfig.yaml
apiVersion: apps.openshift.io/v1
kind: DeploymentConfig
metadata:
  name: sanctuary-broker
  namespace: sanctuary
spec:
  replicas: 3
  selector:
    app: sanctuary-broker
  template:
    metadata:
      labels:
        app: sanctuary-broker
    spec:
      containers:
      - name: gateway
        image: image-registry.openshift-image-registry.svc:5000/sanctuary/sanctuary-broker:latest
        ports:
        - containerPort: 9000
        env:
        - name: PROJECT_ROOT
          value: "/app"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
  triggers:
  - type: ConfigChange
  - type: ImageChange
    imageChangeParams:
      automatic: true
      containerNames:
      - gateway
      from:
        kind: ImageStreamTag
        name: sanctuary-broker:latest
---
apiVersion: v1
kind: Route
metadata:
  name: sanctuary-broker
  namespace: sanctuary
spec:
  to:
    kind: Service
    name: sanctuary-broker
  port:
    targetPort: 9000
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
```

**Deploy to OpenShift:**
```bash
# Login to OpenShift
oc login https://api.openshift.example.com

# Create project
oc new-project sanctuary

# Create image stream
oc create imagestream sanctuary-broker

# Build and push image
oc new-build --binary --name=sanctuary-broker
oc start-build sanctuary-broker --from-dir=. --follow

# Deploy
oc apply -f openshift/

# Expose route
oc expose svc/sanctuary-broker

# Check status
oc get pods
oc get routes
```

---

### 4.5 Container Runtime Decision Matrix

| Criteria | Podman | Docker | Kubernetes | OpenShift |
|----------|--------|--------|------------|----------|
| **Ease of Setup** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Security** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Scalability** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | Free | Free | Free (OSS) | $$$ |
| **Learning Curve** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Ecosystem** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ops Overhead** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Best For** | Local/Dev | Local/Dev | Cloud/Prod | Enterprise |

**Recommendation by Scale:**
- **1-5 servers:** Podman or Docker (single host)
- **5-20 servers:** Podman with systemd (single host)
- **20-100 servers:** Kubernetes (multi-host)
- **100+ servers:** Kubernetes or OpenShift (enterprise)

**Current Sanctuary Scale:** 12 servers → **Podman** (with K8s migration path)

---

### 4.6 Migration Path

**Phase 1: Local Development (Week 1-2)**
- Runtime: Podman Compose
- Servers: 3 (rag_cortex, task, git_workflow)
- Purpose: MVP validation

**Phase 2: Single-Host Production (Week 3-4)**
- Runtime: Podman with systemd
- Servers: 12 (all servers)
- Purpose: Production deployment

**Phase 3: Multi-Host Scale (Month 2+)**
- Runtime: Kubernetes
- Servers: 12+ (with auto-scaling)
- Purpose: Cloud deployment, high availability

**Migration Commands:**
```bash
# Phase 1 → Phase 2 (Podman Compose → systemd)
podman generate systemd --name sanctuary-broker-mcp > /etc/systemd/system/sanctuary-broker.service
systemctl enable --now sanctuary-broker

# Phase 2 → Phase 3 (Podman → Kubernetes)
kubectl create namespace sanctuary
kubectl apply -f kubernetes/
```

---

## 5. Migration Strategy

### 5.1 Side-by-Side Deployment

**Approach:** Run both old and new architectures simultaneously

**Benefits:**
- Zero downtime
- Easy comparison
- Safe rollback
- Gradual migration

**Implementation:**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "🆕 Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.gateway.server"],
      "env": { ... }
    },
    "rag_cortex_legacy": {
      "displayName": "📦 RAG Cortex (Legacy)",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.rag_cortex.server"],
      "env": { ... }
    }
    // ... other legacy servers
  }
}
```

**Testing Workflow:**
1. Call tool through gateway: `sanctuary-broker.cortex_query(...)`
2. Call tool directly: `rag_cortex_legacy.cortex_query(...)`
3. Compare responses
4. Measure latency difference

### 5.2 Gradual Migration

**Week 1:** Gateway + 3 servers (rag_cortex, task, git_workflow)  
**Week 2:** Gateway + 6 servers (add adr, chronicle, protocol)  
**Week 3:** Gateway + 9 servers (add council, persona, forge)  
**Week 4:** Gateway + 12 servers (add config, code, orchestrator)

### 5.3 Rollback Plan

**If Gateway Fails:**
1. Revert `claude_desktop_config.json` to backup
2. Restart Claude Desktop
3. All legacy servers still work
4. No data loss

**Recovery Time:** <5 minutes

---

## 6. Documentation Structure

### 6.1 Proposed Directory Structure

```
docs/
├── mcp/                        # Existing MCP docs
│   ├── architecture.md         # Update with Gateway section
│   ├── QUICKSTART.md           # Update with Gateway instructions
│   └── servers/                # Existing server docs (unchanged)
│       ├── rag_cortex/
│       ├── git/
│       └── ...
└── mcp_gateway/                # NEW: Gateway-specific docs
    ├── README.md               # Overview and quick start
    ├── ARCHITECTURE.md         # Architecture deep dive
    ├── OPERATIONS.md           # Operations reference
    ├── DEPLOYMENT.md           # Deployment guide
    ├── TESTING.md              # Testing guide
    ├── TROUBLESHOOTING.md      # Common issues
    ├── MIGRATION.md            # Migration guide
    └── diagrams/               # Architecture diagrams
        ├── deployment.mmd      # Deployment diagram
        ├── request_flow.mmd    # Request flow diagram
        └── registry_schema.mmd # Registry schema diagram
```

### 6.2 Documentation Deliverables

**docs/mcp_gateway/ARCHITECTURE.md:**
- Gateway architecture overview
- Component descriptions
- Design decisions
- Trade-offs

**docs/mcp_gateway/OPERATIONS.md:**
- All Gateway operations (tools)
- Registry management
- Health checks
- Monitoring

**docs/mcp_gateway/DEPLOYMENT.md:**
- Deployment modes
- Container setup
- Configuration
- Environment variables

**docs/mcp_gateway/TESTING.md:**
- Unit test guide
- Integration test guide
- E2E test guide
- Performance test guide

**docs/mcp_gateway/TROUBLESHOOTING.md:**
- Common issues
- Debugging tips
- Log analysis
- Recovery procedures

**docs/mcp_gateway/MIGRATION.md:**
- Migration checklist
- Side-by-side deployment
- Rollback procedures
- Validation steps

---

## 7. Success Metrics

### 7.1 Phase 1 (MVP) Success Criteria

- [ ] Gateway routes to 3 backend servers
- [ ] All 25 tools work correctly
- [ ] Latency overhead <50ms
- [ ] No functionality regressions
- [ ] Unit tests pass (80%+ coverage)

### 7.2 Phase 2 (Registry) Success Criteria

- [ ] Registry database created and populated
- [ ] Router uses registry for all lookups
- [ ] Health checks run every 60 seconds
- [ ] Audit log captures all tool calls
- [ ] Integration tests pass

### 7.3 Phase 3 (Full Migration) Success Criteria

- [ ] All 12 servers registered
- [ ] All 63 tools accessible through gateway
- [ ] All integration tests pass
- [ ] Latency overhead <50ms
- [ ] No functionality regressions
- [ ] Side-by-side deployment working

### 7.4 Phase 4 (Production) Success Criteria

- [ ] Security allowlist enforced
- [ ] Metrics dashboard available
- [ ] Circuit breaker prevents cascading failures
- [ ] Cache hit rate >50% for read operations
- [ ] All documentation updated
- [ ] E2E tests pass

---

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| Gateway becomes bottleneck | Connection pooling, async I/O, benchmarking |
| Latency overhead unacceptable | Early benchmarking, optimization, caching |
| Dynamic tool registration fails | Fallback to static registration |
| Backend compatibility issues | No backend changes required |
| SQLite corruption | Regular backups, transaction safety |

### 8.2 Operational Risks

| Risk | Mitigation |
|------|------------|
| Gateway crashes | Auto-restart, health checks, monitoring |
| Configuration errors | Schema validation, dry-run mode |
| Migration breaks workflows | Side-by-side deployment, gradual migration |
| User confusion | Clear documentation, training |

---

## 9. Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Planning | 1 week | Documentation, ADR, Protocol |
| Phase 2: MVP | 1-2 days | Gateway + 3 servers |
| Phase 3: Registry | 1-2 days | SQLite registry, health checks |
| Phase 4: Full Migration | 2-3 days | All 12 servers migrated |
| Phase 5: Production | 3-4 days | Security, monitoring, optimization |
| **Total** | **4-6 weeks** | **Production-ready Gateway** |

---

## 10. Next Steps

### Immediate (This Week)
1. ✅ Complete research (done)
2. ✅ Create ADR 056 (done)
3. **Create Protocol 122:** Dynamic Server Binding
4. **Create Architecture Docs:** `docs/mcp_gateway/ARCHITECTURE.md`
5. **Create Operations Reference:** `docs/mcp_gateway/OPERATIONS.md`

### Short-Term (Week 2)
1. **Build Gateway MVP:** 3 servers
2. **Test with Claude Desktop:** Validate all workflows
3. **Implement Registry:** SQLite database

### Medium-Term (Week 3-4)
1. **Migrate All Servers:** Add remaining 9 servers
2. **Side-by-Side Deployment:** Test both architectures
3. **Full Integration Testing:** Validate all 63 tools

### Long-Term (Week 5-6)
1. **Production Hardening:** Security, monitoring, optimization
2. **Documentation:** Update all docs
3. **Training:** Team onboarding

---

## 11. Appendix

### A. Full Tool Registry SQL

```sql
-- All 63 tools across 12 servers
-- (See separate file: mcp_servers/gateway/config/populate_registry.sql)
```

### B. Gateway API Specification

```yaml
# OpenAPI specification for Gateway management API
# (See separate file: docs/mcp_gateway/api_spec.yaml)
```

### C. Performance Benchmarks

```
# Baseline measurements
# (See separate file: docs/mcp_gateway/benchmarks.md)
```

---

## Conclusion

This implementation plan provides a **comprehensive roadmap** for migrating to the Dynamic MCP Gateway Architecture. The phased approach minimizes risk while delivering incremental value.

**Key Takeaways:**
- **Low Risk:** Side-by-side deployment, easy rollback
- **Incremental:** 5 phases, each with clear deliverables
- **Measurable:** Success criteria for each phase
- **Documented:** Comprehensive documentation plan

**Recommendation:** Proceed with Phase 1 (Planning & Documentation) immediately.
