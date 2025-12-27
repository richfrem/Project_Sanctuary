# MCP Fleet Quick Fix - Execution Script

**Target:** Fix unhealthy containers and ChromaDB connectivity  
**Time:** 30-45 minutes  
**Risk:** LOW (changes are in docker-compose.yml only)

---

## Pre-Flight Checks

```bash
# 1. Check current container status
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Expected: 5 unhealthy containers, 1 with no health status
# sanctuary_utils (unhealthy)
# sanctuary_filesystem (unhealthy)
# sanctuary_network (unhealthy)
# sanctuary_git (unhealthy)
# sanctuary_cortex (unhealthy)
# sanctuary_domain (Up - no healthcheck)

# 2. Verify ChromaDB is actually running
podman exec sanctuary_vector_db curl -s http://localhost:8000/api/v1/heartbeat

# Expected: {"nanosecond heartbeat": <number>}

# 3. Try to connect from cortex (will fail)
podman exec sanctuary_cortex curl -s http://sanctuary_vector_db:8000/api/v1/heartbeat

# Expected: curl: (6) Could not resolve host: sanctuary_vector_db
# (Because CHROMA_HOST is set to "vector_db" not "sanctuary_vector_db")
```

---

## Fix 1: Update docker-compose.yml

**File:** `/Users/richardfremmerlid/Projects/Project_Sanctuary/docker-compose.yml`

### Change 1: Fix CHROMA_HOST (Line ~140)

**Find:**
```yaml
  sanctuary_cortex:
    ...
    environment:
      ...
      - CHROMA_HOST=vector_db
```

**Replace with:**
```yaml
  sanctuary_cortex:
    ...
    environment:
      ...
      - CHROMA_HOST=sanctuary_vector_db
```

### Change 2: Fix healthcheck for sanctuary_utils (Line ~70)

**Find:**
```yaml
  sanctuary_utils:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/sse" ]
```

**Replace with:**
```yaml
  sanctuary_utils:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health" ]
```

### Change 3: Fix healthcheck for sanctuary_filesystem (Line ~95)

**Find:**
```yaml
  sanctuary_filesystem:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/sse" ]
```

**Replace with:**
```yaml
  sanctuary_filesystem:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health" ]
```

### Change 4: Fix healthcheck for sanctuary_network (Line ~121)

**Find:**
```yaml
  sanctuary_network:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/sse" ]
```

**Replace with:**
```yaml
  sanctuary_network:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health" ]
```

### Change 5: Fix healthcheck for sanctuary_git (Line ~146)

**Find:**
```yaml
  sanctuary_git:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/sse" ]
```

**Replace with:**
```yaml
  sanctuary_git:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health" ]
```

### Change 6: Fix healthcheck for sanctuary_cortex (Line ~174)

**Find:**
```yaml
  sanctuary_cortex:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/sse" ]
```

**Replace with:**
```yaml
  sanctuary_cortex:
    ...
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health" ]
```

### Change 7: Add healthcheck for sanctuary_domain (after Line ~209)

**Find:**
```yaml
  sanctuary_domain:
    ...
    command: [ "python", "-m", "mcp_servers.gateway.clusters.sanctuary_domain.server" ]
    restart: unless-stopped
```

**Add BEFORE restart line:**
```yaml
  sanctuary_domain:
    ...
    command: [ "python", "-m", "mcp_servers.gateway.clusters.sanctuary_domain.server" ]
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8105/health" ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
```

---

## Fix 2: Ensure /health endpoints exist

All servers should have a `/health` endpoint. Let me verify the SSEServer provides this:

**File:** `mcp_servers/lib/sse_adaptor.py` (check if /health route exists)

If missing, we need to add it. But based on the code in sanctuary_utils/server.py, SSEServer should expose FastAPI's app which typically has health endpoints.

**Alternative:** If /health doesn't exist, we can use a simpler healthcheck:

```yaml
healthcheck:
  test: [ "CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/sse').close()" ]
```

---

## Deploy Changes

```bash
# 1. Stop all containers
cd /Users/richardfremmerlid/Projects/Project_Sanctuary
podman compose down

# 2. Verify docker-compose.yml changes are saved

# 3. Rebuild ALL containers (to pick up any code changes)
podman compose build

# 4. Start all containers
podman compose up -d

# 5. Wait for startup (60 seconds)
echo "Waiting for containers to stabilize..."
sleep 60

# 6. Check status
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Expected: All containers showing "healthy" or "Up" (not "unhealthy")
```

---

## Validation Tests

### Test 1: ChromaDB Connectivity from Cortex

```bash
podman exec sanctuary_cortex curl -s http://sanctuary_vector_db:8000/api/v1/heartbeat

# Expected: {"nanosecond heartbeat": <timestamp>}
# If you get "Could not resolve host", CHROMA_HOST fix didn't apply
```

### Test 2: Health Endpoints

```bash
# Test each container's health endpoint
for port in 8100 8101 8102 8103 8104; do
  echo "Testing port $port..."
  curl -s http://localhost:$port/health && echo " ✓ OK" || echo " ✗ FAIL"
done

# sanctuary_domain uses port 8105 for BOTH internal and external
curl -s http://localhost:8105/health && echo "8105 ✓ OK" || echo "8105 ✗ FAIL"
```

### Test 3: SSE Endpoints (should still work)

```bash
# These should return SSE streams (first few lines)
for port in 8100 8101 8102 8103 8104 8105; do
  echo "=== Testing SSE on port $port ==="
  timeout 2 curl -N -s http://localhost:$port/sse | head -n 4
  echo ""
done

# Expected output for each:
# event: endpoint
# data: /messages
# (blank line)
# event: ping
```

### Test 4: Gateway Tool Discovery

Use the Gateway admin UI: http://localhost:4444/admin/#tools

**Expected tool counts:**
- sanctuary_utils: 16 tools
- sanctuary_filesystem: 9 tools
- sanctuary_network: 2 tools
- sanctuary_git: 9 tools
- sanctuary_cortex: 13 tools
- sanctuary_domain: 35 tools
- **Total: 84 tools**

### Test 5: RAG Cortex Operations

```bash
# Use MCP tools to test
# sanctuary_gateway:cortex-cortex-get-stats
# Expected: Returns statistics about the vector database

# sanctuary_gateway:cortex-cortex-capture-snapshot
# Expected: Creates snapshot without "Could not connect to Chroma" error
```

---

## Success Criteria

- ✅ **All 6 containers show "healthy" status** (not "unhealthy")
- ✅ **ChromaDB connectivity works** from sanctuary_cortex
- ✅ **All health endpoints respond** (8100, 8101, 8102, 8103, 8104, 8105)
- ✅ **All SSE endpoints stream properly** (event: endpoint)
- ✅ **Gateway discovers all 84 tools**
- ✅ **cortex_get_stats returns data** (no connection errors)
- ✅ **cortex_capture_snapshot works** (no ChromaDB errors)

---

## Rollback Plan

If anything goes wrong:

```bash
# 1. Stop all containers
podman compose down

# 2. Revert docker-compose.yml changes
git checkout HEAD -- docker-compose.yml

# 3. Restart with old config
podman compose up -d

# 4. Report issues for further investigation
```

---

## If /health Endpoints Don't Exist

**Alternative Fix:** Instead of changing healthcheck to use /health, we can change it to just check if the port is open:

```yaml
healthcheck:
  test: [ "CMD", "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8000/sse" ]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

This tests if the SSE endpoint responds (even with an empty body) without failing on the -f flag.

---

## Common Issues

### Issue: "Could not resolve host: sanctuary_vector_db"
**Cause:** CHROMA_HOST still set to "vector_db"  
**Fix:** Verify docker-compose.yml change was saved and containers were rebuilt

### Issue: Containers still showing "unhealthy"
**Cause:** /health endpoint doesn't exist  
**Fix:** Use alternative healthcheck (see above)

### Issue: "Connection refused" on health checks
**Cause:** Server not fully started yet  
**Fix:** Wait longer (increase start_period to 20s)

### Issue: Gateway can't discover tools
**Cause:** SSE handshake failing  
**Fix:** Check SSE endpoints manually with curl -N

---

## Post-Fix Actions

1. **Update documentation** with correct port mappings
2. **Create chronicle entry** documenting the fixes
3. **Run Protocol 125 learning loop** to validate full workflow
4. **Monitor logs** for 24 hours to catch any issues
5. **Update TROUBLESHOOTING.md** with these solutions

---

## Notes for Agent Execution

- **Use code MCP tools** to read and modify docker-compose.yml
- **Don't restart containers one-by-one** - do full compose down/up cycle
- **Wait 60 seconds** after compose up before testing (containers need time to stabilize)
- **Test systematically** - ChromaDB first, then health endpoints, then SSE, then Gateway
- **Document any deviations** from this script in a chronicle entry
