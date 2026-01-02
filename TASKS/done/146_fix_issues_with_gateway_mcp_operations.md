# MCP Integration Issues - Comprehensive Remediation Plan

**Date:** 2024-12-24  
**Status:** ✅ ALL PHASES COMPLETE  
**Context:** Post-ADR-066 v1.3 Gateway Integration Issues  
**Target:** Antigravity AI Agent Execution

---

## Executive Summary

The Gateway fleet is experiencing integration failures due to:
1. **ChromaDB connectivity issues** preventing RAG Cortex operations
2. **Incomplete ADR-076 migration** causing tool registration inconsistencies
3. **Legacy MCP server coexistence** creating namespace conflicts
4. **Missing integration test coverage** for SSE transport

---

## Issue Analysis

### Issue 1: ChromaDB Connection Failure ⚠️ CRITICAL
**Symptom:** `Could not connect to a Chroma server`  
**Root Cause:** Incorrect CHROMA_HOST in docker-compose.yml  
**Impact:** All RAG Cortex tools non-functional (13 tools affected)

**Technical Details:**
- ChromaDB container: `sanctuary_vector_db` on port 8110:8000
- docker-compose.yml sets: `CHROMA_HOST=vector_db` ❌ WRONG
- operations.py expects: `sanctuary_vector_db` ✅ CORRECT
- Container name is `sanctuary_vector_db` not `vector_db`
- **Fix:** Change `CHROMA_HOST=vector_db` → `CHROMA_HOST=sanctuary_vector_db`

### Issue 2: Unhealthy Container Healthchecks ⚠️ HIGH
**Symptom:** 5 containers showing "unhealthy" status  
**Root Cause:** Healthcheck expects SSE handshake but curl is getting empty response  
**Impact:** Containers restart repeatedly, Gateway can't maintain stable connections

**Affected Containers:**
- ❌ sanctuary_utils (port 8100) - `curl -f http://localhost:8000/sse` failing
- ❌ sanctuary_network (port 8102) - `curl -f http://localhost:8000/sse` failing
- ❌ sanctuary_git (port 8103) - `curl -f http://localhost:8000/sse` failing
- ❌ sanctuary_filesystem (port 8101) - `curl -f http://localhost:8000/sse` failing
- ❌ sanctuary_cortex (port 8104) - `curl -f http://localhost:8000/sse` failing
- ⚠️ sanctuary_domain (port 8105) - No healthcheck defined

**Technical Details:**
- Healthcheck command: `curl -f http://localhost:8000/sse`
- Expected: SSE stream with `event: endpoint`
- Actual: Empty response or connection refused
- **Root Cause:** SSE endpoint may not be responding properly OR curl is closing connection before stream starts
**Symptom:** Some tools working, others failing in Gateway  
**Root Cause:** Mixed decorator patterns (ADR-076 migration incomplete)  
**Impact:** Tool discovery inconsistencies across fleet

**Status by Server:**
- ✅ **sanctuary_utils** - Fully migrated to `@sse_tool` decorator (16 tools)
- ✅ **sanctuary_network** - Fully migrated to `@sse_tool` decorator (2 tools)
- ✅ **sanctuary_git** - Fully migrated to `@sse_tool` decorator (9 tools)
- ✅ **sanctuary_domain** - Fully migrated to `@sse_tool` decorator (35 tools)
- ❌ **sanctuary_cortex** - OLD PATTERN (register_tool calls) (13 tools)
- ❌ **sanctuary_filesystem** - STATUS UNKNOWN (needs verification)

### Issue 3: Port Mapping Discrepancy 🔧 MEDIUM
**Symptom:** Documentation says cortex on 8105, actually on 8104  
**Root Cause:** Documentation drift from actual deployment  
**Impact:** Confusion, incorrect tool calls, documentation errors

**Actual Port Mapping:**
- sanctuary_utils: 8100:8000 ✅
- sanctuary_filesystem: 8101:8000 ✅
- sanctuary_network: 8102:8000 ✅
- sanctuary_git: 8103:8000 ✅
- **sanctuary_cortex: 8104:8000** ✅ (not 8105)
- **sanctuary_domain: 8105:8105** ✅ (unique - uses 8105 internally AND externally)

### Issue 4: Tool Registration Inconsistency 🔧 LOW
**Symptom:** Two parallel MCP systems (gateway clusters vs. legacy servers)  
**Root Cause:** Incomplete migration from Task 144  
**Impact:** Configuration confusion, duplicate tooling

**Legacy Servers Still Present:**
- `mcp_servers/rag_cortex/` (legacy)
- `mcp_servers/git/` (legacy - but used by gateway)
- `mcp_servers/filesystem/` (legacy)
- Various other legacy servers

**Gateway Clusters:**
- `mcp_servers/gateway/clusters/sanctuary_*` (current)

### Issue 4: Integration Test Gaps
**Symptom:** No E2E validation of Gateway SSE handshake  
**Root Cause:** Tests focus on STDIO, not SSE transport  
**Impact:** Silent failures in production Gateway deployment

---

## Remediation Plan

### Phase 1: Critical Infrastructure Fixes (1 hour)
**Priority:** CRITICAL  
**Goal:** Fix ChromaDB connectivity and container health checks

#### Step 1.1: Verify Current State
```bash
# Check container status
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check ChromaDB
podman exec sanctuary_vector_db curl -s http://localhost:8000/api/v1/heartbeat

# Try to connect from cortex
podman exec sanctuary_cortex curl -s http://sanctuary_vector_db:8000/api/v1/heartbeat
# (This will likely fail with current config)
```

#### Step 1.2: Update ChromaDB Connection Config
**File:** `docker-compose.yml` line 140

**Current (BROKEN):**
```yaml
environment:
  - CHROMA_HOST=vector_db  # ❌ WRONG - container name is sanctuary_vector_db
  - CHROMA_PORT=8000
```

**Fixed:**
```yaml
environment:
  - CHROMA_HOST=sanctuary_vector_db  # ✅ CORRECT - matches container_name
  - CHROMA_PORT=8000
```

**Why this matters:**
- Container name: `sanctuary_vector_db`
- Service name in compose: `vector_db`
- Docker networking uses **container name**, not service name
- operations.py will look for `sanctuary_vector_db:8000`

#### Step 1.3: Fix Healthcheck Commands
**File:** `docker-compose.yml` (all sanctuary_* services)

**Current (BROKEN):**
```yaml
healthcheck:
  test: [ "CMD", "curl", "-f", "http://localhost:8000/sse" ]
```

**Why this fails:**
- `curl` closes connection immediately
- SSE streams need persistent connection
- `-f` flag makes curl exit on HTTP errors (which SSE stream may trigger)

**Fixed Option A - Test /health endpoint instead:**
```yaml
healthcheck:
  test: [ "CMD", "curl", "-f", "http://localhost:8000/health" ]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

**Fixed Option B - Test SSE with proper flags:**
```yaml
healthcheck:
  test: [ "CMD", "timeout", "2", "curl", "-N", "-s", "http://localhost:8000/sse" ]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

**Recommendation:** Use Option A (/health endpoint) - simpler and more reliable

**Note:** sanctuary_domain needs healthcheck added (currently missing)
```bash
# Update environment in docker-compose.yml or Containerfile
# Add: CHROMA_HOST=sanctuary_vector_db

# Rebuild cortex container
podman compose build sanctuary_cortex

# Restart the service
podman compose restart sanctuary_cortex

# Verify connectivity
curl http://localhost:8105/health  # Cortex health check
```

#### Step 1.5: Validation
```bash
# Test ChromaDB connectivity from cortex container
podman exec sanctuary_cortex curl -s http://sanctuary_vector_db:8000/api/v1/heartbeat

# Expected output: {"nanosecond heartbeat": <timestamp>}

# Test all container health endpoints
for port in 8100 8101 8102 8103 8104 8105; do
  echo "Testing port $port..."
  curl -s http://localhost:$port/health || echo "No /health endpoint on $port"
done

# Test SSE endpoints (should stream events)
for port in 8100 8101 8102 8103 8104 8105; do
  echo "Testing SSE on port $port..."
  timeout 2 curl -N -s http://localhost:$port/sse | head -n 4
done

# Test via Gateway tools
# Use sanctuary_gateway:cortex-cortex-get-stats
# Use sanctuary_gateway:cortex-cortex-capture-snapshot
```

**Success Criteria:**
- ✅ All containers show "healthy" status (not "unhealthy")
- ✅ ChromaDB heartbeat responds successfully
- ✅ All SSE endpoints return `event: endpoint` header
- ✅ `cortex_get_stats` returns database statistics
- ✅ `cortex_capture_snapshot` completes without connection errors

---

### Phase 2: Standardize Cortex Server on ADR-076 (1 hour)
**Priority:** HIGH  
**Goal:** Migrate `sanctuary_cortex` to `@sse_tool` decorator pattern

#### Step 2.1: Identify Current Pattern
**File:** `mcp_servers/gateway/clusters/sanctuary_cortex/server.py`

**Current (OLD):**
```python
server.register_tool("cortex_ingest_full", cortex_ingest_full, INGEST_FULL_SCHEMA)
server.register_tool("cortex_query", cortex_query, QUERY_SCHEMA)
# ... 11 more
```

#### Step 2.2: Refactor to Decorator Pattern
**Target Pattern (matching sanctuary_utils):**
```python
from mcp_servers.lib.sse_adaptor import SSEServer, sse_tool

server = SSEServer("sanctuary_cortex", version="1.0.0")

@sse_tool(
    name="cortex_ingest_full",
    description="Perform full re-ingestion of the knowledge base.",
    schema=INGEST_FULL_SCHEMA
)
def cortex_ingest_full(purge_existing: bool = False, source_directories: List[str] = None):
    response = get_ops().ingest_full(purge_existing=purge_existing, source_directories=source_directories)
    return json.dumps(to_dict(response), indent=2)

# ... repeat for all 13 tools

# Auto-register all decorated tools (ADR-076)
server.register_decorated_tools(locals())
```

#### Step 2.3: Testing Strategy
```bash
# 1. Unit test: Verify tool schemas match
pytest tests/integration/test_sanctuary_cortex_server.py -k schema

# 2. Integration test: Verify SSE handshake
curl -N http://localhost:8105/sse
# Expected: event: endpoint\ndata: /messages

# 3. E2E test: Verify Gateway tool discovery
python -m mcp_servers.gateway.verify_hello_world_rpc --server cortex
```

#### Step 2.4: Commit and Deploy
```bash
git add mcp_servers/gateway/clusters/sanctuary_cortex/server.py
git commit -m "feat(cortex): Migrate to ADR-076 @sse_tool pattern

- Replace register_tool() calls with @sse_tool decorators
- Add server.register_decorated_tools(locals())
- Maintain STDIO/SSE dual-transport compliance
- Follows sanctuary_utils reference pattern

Ref: ADR-076, Task-144"

podman compose build sanctuary_cortex
podman compose restart sanctuary_cortex
```

**Success Criteria:**
- ✅ All 13 Cortex tools discoverable in Gateway admin UI
- ✅ SSE handshake passes `curl -N /sse` test
- ✅ Integration tests pass

---

### Phase 3: Verify Filesystem Server (30 min)
**Priority:** MEDIUM  
**Goal:** Confirm sanctuary_filesystem ADR-076 compliance

#### Step 3.1: Inspect Server Implementation
```bash
# Check for decorator pattern usage
grep -n "@sse_tool" mcp_servers/gateway/clusters/sanctuary_filesystem/server.py

# If missing, check for old pattern
grep -n "register_tool" mcp_servers/gateway/clusters/sanctuary_filesystem/server.py
```

#### Step 3.2: Apply Same Refactor (if needed)
- Follow Phase 2 steps if using old pattern
- Otherwise, verify tool discovery in Gateway

#### Step 3.3: Integration Test
```bash
# Verify filesystem tools are accessible
python -m mcp_servers.gateway.verify_hello_world_rpc --server filesystem

# Check Gateway admin UI for tool count
# Expected: 9 tools (read, write, list, find, search, info, lint, format, analyze)
```

**Success Criteria:**
- ✅ All filesystem tools listed in Gateway `/admin/#tools`
- ✅ Sample file operation succeeds via Gateway

---

### Phase 4: Legacy Server Deprecation Strategy (2 hours)
**Priority:** LOW (can defer)  
**Goal:** Clean up parallel MCP architectures

#### Step 4.1: Audit Legacy vs. Gateway Usage
**Create inventory:**
```markdown
| Legacy Server | Gateway Equivalent | Status | Action |
|---------------|-------------------|--------|--------|
| mcp_servers/rag_cortex/ | sanctuary_cortex | ⚠️ Shared ops | Document as library |
| mcp_servers/git/ | sanctuary_git | ⚠️ Shared ops | Document as library |
| mcp_servers/filesystem/ | sanctuary_filesystem | ❓ Unknown | Investigate |
| mcp_servers/chronicle/ | sanctuary_domain | ✅ Replaced | Archive |
| mcp_servers/protocol/ | sanctuary_domain | ✅ Replaced | Archive |
| mcp_servers/task/ | sanctuary_domain | ✅ Replaced | Archive |
| mcp_servers/adr/ | sanctuary_domain | ✅ Replaced | Archive |
```

#### Step 4.2: Clarify Architecture Pattern
**Recommendation:** Legacy servers become **operations libraries only**

```
mcp_servers/
├── gateway/
│   └── clusters/          # MCP server implementations (dual-transport)
│       ├── sanctuary_cortex/
│       │   └── server.py  # Uses mcp_servers/rag_cortex/operations.py
│       └── sanctuary_git/
│           └── server.py  # Uses mcp_servers/git/operations.py
├── rag_cortex/
│   └── operations.py      # Business logic ONLY (no MCP server)
└── git/
    └── operations.py      # Business logic ONLY (no MCP server)
```

#### Step 4.3: Update Documentation
- **README.md:** Clarify that `mcp_servers/gateway/clusters/` are the ONLY server entry points
- **ADR-066:** Add note about operations libraries vs. server implementations
- **Protocol 101:** Update with new architecture clarity

**Deferred Actions (not urgent):**
- Remove legacy `server.py` files from non-gateway locations
- Consolidate Dockerfiles into `mcp_servers/gateway/clusters/*/Containerfile`

---

### Phase 5: Integration Test Hardening (2 hours)
**Priority:** MEDIUM  
**Goal:** Prevent future silent failures

#### Step 5.1: Add SSE Handshake Tests
**File:** `tests/integration/test_sse_handshake.py` (NEW)

```python
import pytest
import asyncio
import httpx

SERVERS = [
    ("sanctuary_utils", 8100),
    ("sanctuary_network", 8101),
    ("sanctuary_git", 8102),
    ("sanctuary_filesystem", 8103),
    ("sanctuary_cortex", 8105),
    ("sanctuary_domain", 8104),
]

@pytest.mark.parametrize("server_name,port", SERVERS)
def test_sse_handshake(server_name, port):
    """Verify SSE endpoint returns proper handshake."""
    with httpx.stream("GET", f"http://localhost:{port}/sse", timeout=5.0) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        
        # Read first event
        line = next(response.iter_lines())
        assert line == "event: endpoint", f"{server_name} did not send endpoint event"
        
        line = next(response.iter_lines())
        assert line == "data: /messages", f"{server_name} endpoint data incorrect"
```

#### Step 5.2: Add Gateway Tool Discovery Tests
**File:** `tests/e2e/test_gateway_discovery.py` (NEW)

```python
import pytest
import httpx

EXPECTED_TOOL_COUNTS = {
    "sanctuary_utils": 16,
    "sanctuary_network": 2,
    "sanctuary_git": 9,
    "sanctuary_filesystem": 9,
    "sanctuary_cortex": 13,
    "sanctuary_domain": 35,
}

def test_gateway_tool_discovery():
    """Verify Gateway discovers all expected tools from fleet."""
    response = httpx.get("http://localhost:4444/api/tools")
    assert response.status_code == 200
    
    tools = response.json()
    tool_counts = {}
    
    for tool in tools:
        server = tool["server_name"]
        tool_counts[server] = tool_counts.get(server, 0) + 1
    
    for server, expected_count in EXPECTED_TOOL_COUNTS.items():
        actual_count = tool_counts.get(server, 0)
        assert actual_count == expected_count, \
            f"{server}: expected {expected_count} tools, got {actual_count}"
```

#### Step 5.3: Add to CI Pipeline
**File:** `.github/workflows/integration_tests.yml` (UPDATE)

```yaml
- name: Run SSE Handshake Tests
  run: pytest tests/integration/test_sse_handshake.py -v

- name: Run Gateway Discovery Tests
  run: pytest tests/e2e/test_gateway_discovery.py -v
```

**Success Criteria:**
- ✅ All SSE handshake tests pass
- ✅ Gateway discovers all 84 tools (sum of expected counts)
- ✅ Tests run in CI on every PR

---

### Phase 6: Documentation and Validation (1 hour)
**Priority:** MEDIUM  
**Goal:** Ensure reproducibility and knowledge transfer

#### Step 6.1: Update Operations Inventory
**File:** `docs/architecture/mcp/OPERATIONS_INVENTORY.md` (UPDATE)

Add section:
```markdown
## Gateway Fleet Status (Post-ADR-066 v1.3)

| Server | Port | Tools | ADR-076 Compliant | ChromaDB Deps | Status |
|--------|------|-------|-------------------|---------------|--------|
| sanctuary_utils | 8100 | 16 | ✅ Yes | ❌ No | ✅ Operational |
| sanctuary_network | 8101 | 2 | ✅ Yes | ❌ No | ✅ Operational |
| sanctuary_git | 8102 | 9 | ✅ Yes | ❌ No | ✅ Operational |
| sanctuary_filesystem | 8103 | 9 | ✅ Yes | ❌ No | ✅ Operational |
| sanctuary_cortex | 8105 | 13 | ✅ Yes | ✅ Yes | 🟡 Fixed in Phase 1/2 |
| sanctuary_domain | 8104 | 35 | ✅ Yes | ❌ No | ✅ Operational |

**Total Gateway Tools:** 84
```

#### Step 6.2: Create Troubleshooting Guide
**File:** `docs/TROUBLESHOOTING.md` (NEW)

```markdown
# MCP Gateway Troubleshooting

## Issue: "Could not connect to a Chroma server"

**Symptom:** RAG Cortex tools fail with ChromaDB connection error

**Diagnosis:**
1. Verify ChromaDB container is running: `podman ps | grep vector_db`
2. Check container networking: `podman inspect sanctuary_cortex | grep Networks`
3. Test connectivity from cortex: `podman exec sanctuary_cortex curl http://sanctuary_vector_db:8000/api/v1/heartbeat`

**Fix:**
Ensure `CHROMA_HOST=sanctuary_vector_db` in cortex container environment.

## Issue: Tools not appearing in Gateway admin UI

**Symptom:** Expected tools missing from `/admin/#tools`

**Diagnosis:**
1. Check SSE handshake: `curl -N http://localhost:PORT/sse`
2. Verify ADR-076 compliance: `grep "@sse_tool" server.py`
3. Check logs: `podman logs sanctuary_<server>`

**Fix:**
Migrate to `@sse_tool` decorator pattern per ADR-076 (see Phase 2).
```

#### Step 6.3: Chronicle Entry
**Action:** Create chronicle entry documenting remediation

```bash
# Use sanctuary_gateway:domain-chronicle-create-entry
Title: "MCP Integration Remediation Complete (ADR-066 Post-Mortem)"
Content: """
Completed systematic remediation of Gateway integration issues discovered post-ADR-066 v1.3 deployment.

## Issues Resolved:
1. ChromaDB container networking (CHROMA_HOST=sanctuary_vector_db)
2. Cortex server migration to ADR-076 @sse_tool pattern
3. Filesystem server ADR-076 compliance verification
4. Integration test coverage gaps

## Validation:
- All 84 Gateway tools now discoverable
- SSE handshake tests passing for all 6 servers
- RAG Cortex snapshot functionality operational

## Follow-up:
- Legacy server deprecation strategy documented (Phase 4)
- TROUBLESHOOTING.md created for future issues

Ref: ADR-066 v1.3, ADR-076, Task-144
"""
```

**Success Criteria:**
- ✅ Documentation updated and committed
- ✅ Chronicle entry created
- ✅ Team can reproduce fixes

---

## Execution Checklist for Antigravity Agent

### Prerequisites
- [ ] Access to sanctuary_gateway MCP tools
- [ ] Access to code MCP tools for file modifications
- [ ] Access to git MCP tools for commits
- [ ] Podman access for container operations

### Phase 1: ChromaDB Fix (30 min)
- [ ] Verify container networking configuration
- [ ] Update CHROMA_HOST environment variable
- [ ] Rebuild and restart sanctuary_cortex container
- [ ] Test cortex_get_stats tool
- [ ] Test cortex_capture_snapshot tool

### Phase 2: Cortex Server Migration (1 hour)
- [ ] Read current sanctuary_cortex/server.py
- [ ] Refactor to @sse_tool decorator pattern
- [ ] Add server.register_decorated_tools(locals())
- [ ] Rebuild container
- [ ] Test SSE handshake with curl
- [ ] Verify tool discovery in Gateway admin UI
- [ ] Commit changes with proper message

### Phase 3: Filesystem Verification (30 min)
- [ ] Inspect sanctuary_filesystem/server.py
- [ ] Apply Phase 2 refactor if needed
- [ ] Test filesystem tools via Gateway
- [ ] Commit changes if modified

### Phase 4: Legacy Deprecation (DEFER)
- [ ] Create inventory of legacy vs. gateway servers
- [ ] Document architecture pattern
- [ ] Update README.md
- [ ] (Optional) Remove legacy server.py files

### Phase 5: Integration Tests (2 hours)
- [ ] Create test_sse_handshake.py
- [ ] Create test_gateway_discovery.py
- [ ] Run tests locally
- [ ] Add to CI pipeline
- [ ] Commit test suite

### Phase 6: Documentation (1 hour)
- [ ] Update OPERATIONS_INVENTORY.md
- [ ] Create TROUBLESHOOTING.md
- [ ] Create chronicle entry
- [ ] Commit documentation

---

## Success Metrics

### Immediate (Phase 1-2)
- ✅ 84/84 tools discoverable in Gateway admin UI
- ✅ 0 ChromaDB connection errors in logs
- ✅ All SSE handshake tests passing

### Short-term (Phase 3-5)
- ✅ 6/6 servers ADR-076 compliant
- ✅ Integration test suite covers SSE transport
- ✅ CI pipeline validates Gateway integration

### Long-term (Phase 6+)
- ✅ Clear architecture documentation
- ✅ Troubleshooting guide prevents repeat issues
- ✅ Legacy server deprecation roadmap defined

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ChromaDB fix breaks existing queries | Low | High | Test with cortex_query before/after |
| Decorator refactor introduces regressions | Medium | Medium | Run integration tests before commit |
| Container rebuild breaks unrelated services | Low | Medium | Rebuild one service at a time |
| Documentation becomes stale | Medium | Low | Add to PR checklist |

---

## Rollback Plan

### If Phase 1 fails:
```bash
# Revert CHROMA_HOST change
podman compose down sanctuary_cortex
# Restore previous Containerfile
podman compose build sanctuary_cortex
podman compose up -d sanctuary_cortex
```

### If Phase 2 fails:
```bash
# Revert server.py changes
git checkout HEAD -- mcp_servers/gateway/clusters/sanctuary_cortex/server.py
podman compose build sanctuary_cortex
podman compose restart sanctuary_cortex
```

---

## Next Steps After Completion

1. **Monitor Gateway logs** for 24 hours post-deployment
2. **Run Protocol 125 learning loop** to validate full workflow
3. **Execute Task 144 completion validation** (standardize all servers on FastMCP STDIO + SSEServer SSE)
4. **Schedule legacy server deprecation** as separate task
5. **Consider ADR-067** if FastMCP releases SSE-compatible version

---

## References

- **ADR-066 v1.3:** MCP Server Transport Standards
- **ADR-076:** SSE Tool Metadata Decorator Pattern
- **Protocol 125:** Autonomous AI Learning System Architecture
- **Protocol 128:** Cognitive Continuity & The Red Team Gate
- **Task 144:** Standardize all MCP servers on FastMCP

---

**Plan Status:** READY FOR EXECUTION  
**Estimated Total Time:** 7 hours (6 hours active, 1 hour monitoring)  
**Recommended Start:** Immediate (ChromaDB fix is blocking critical functionality)
