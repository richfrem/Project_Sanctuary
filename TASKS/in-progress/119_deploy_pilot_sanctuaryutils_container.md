# TASK: Deploy Pilot: sanctuary-utils Container

**Status:** in-progress
**Priority:** Critical
**Lead:** Unassigned
**Dependencies:** Task 118 (Red Team Analysis), ADR 060 (Hybrid Fleet)
**Related Documents:** ADR 060 (Hybrid Fleet), ADR 058 (Gateway Decoupling), docker-compose.yml

---

## 1. Objective

Deploy the first Hybrid Fleet cluster (sanctuary-utils) as a pilot to validate the **Fleet of 7** architecture (ADR 060). Create a single container serving multiple low-risk tools via one SSE endpoint.

## 2. Deliverables

1. Dockerfile for sanctuary-utils (with multi-stage dev/prod)
2. FastAPI application with SSE endpoint and multi-tool routing
3. docker-compose.yml entry for sanctuary-utils
4. Gateway registration script/documentation
5. Test demonstrating successful tool call through gateway
6. README documenting the utils cluster architecture

## 3. Acceptance Criteria

- sanctuary-utils container successfully built and running
- Container exposes SSE endpoint at http://sanctuary-utils:8000/sse
- Gateway successfully registers the utils cluster
- Gateway can call 'What time is it?' and receive valid response
- Hot reloading works in development mode
- Multi-tool routing implemented (time, calculator, uuid)

## Notes

**CRITICAL REQUIREMENT (Red Team Mandated):**
This task MUST implement **Guardrail 2: Dynamic Self-Registration**.
**On Container Startup:**
- Container must POST tool manifest to Gateway
- Must use Docker Network Alias: `http://sanctuary-gateway:4444/api/servers/register`
- Must include: server_name, endpoint, tools list, version, health_check path
**Guardrail 1: Fault Containment:**
- Entry point must handle exceptions gracefully
- A calculator crash must NOT crash the time tool
**Guardrail 3: Network Addressing:**
- Use only Docker Network Aliases (e.g., `http://sanctuary-utils:8000`)
- NEVER use localhost or hardcoded IPs
**Tools to Include:**
- Time (current time, timezone info)
- Calculator (basic math operations)
- UUID (generate UUIDs)
- String utilities (upper, lower, trim)
**Technical Stack:**
- Python 3.11
- FastAPI for SSE endpoint
- Uvicorn with hot reload for dev
- Multi-stage Dockerfile (dev/prod)
**Success Metric:**
Gateway successfully routes a tool call to sanctuary-utils and returns a valid response within 100ms.
**Reference:** ADR 060 - Fleet of 7 Architecture (Grok-approved with sanctuary-cortex fix)

**Status Change (2025-12-17):** backlog → in-progress
Phase transition: Architecture complete, beginning implementation. Updated to Fleet of 7 per Grok's sanctuary-cortex fix.

---

## 4. Implementation Plan (Incremental/Agile)

> **Principle:** Build and test incrementally. Each step must be verified before proceeding.

### Phase 1: Minimal Viable Container (Single Tool)
**Goal:** Get ONE tool working end-to-end before adding complexity.

- [ ] **Step 1.1:** Create directory structure
  - `mcp_servers/utils/` with `__init__.py`, `server.py`, `tools/`
  - `mcp_servers/utils/tools/time_tool.py` (single tool)
  
- [ ] **Step 1.2:** Implement minimal FastAPI SSE server
  - Single `/sse` endpoint
  - Only the Time tool (get current time)
  - Test locally: `python -m mcp_servers.utils.server`
  
- [ ] **Step 1.3:** Create minimal Dockerfile
  - Single-stage first (keep simple)
  - Build and run: `podman build -t sanctuary-utils . && podman run -p 8000:8000 sanctuary-utils`
  - **VERIFY:** curl http://localhost:8000/health returns 200

- [ ] **Step 1.4:** Add to docker-compose.yml (dev profile)
  - Add `sanctuary-utils` service
  - Test: `podman compose up sanctuary-utils`
  - **VERIFY:** Container starts and health check passes

### Phase 2: Multi-Tool Routing (Fault Containment)
**Goal:** Add remaining tools with proper isolation.

- [ ] **Step 2.1:** Add Calculator tool
  - `mcp_servers/utils/tools/calculator_tool.py`
  - Implement Guardrail 1 (try/except per tool)
  - **VERIFY:** Calculator crash doesn't kill Time tool

- [ ] **Step 2.2:** Add UUID tool
  - `mcp_servers/utils/tools/uuid_tool.py`
  - **VERIFY:** All 3 tools accessible via SSE

- [ ] **Step 2.3:** Add String utilities
  - `mcp_servers/utils/tools/string_tool.py`
  - **VERIFY:** 4 tools working, fault isolation tested

### Phase 3: Gateway Integration
**Goal:** Connect to external Gateway (IBM ContextForge).

- [ ] **Step 3.1:** Implement Self-Registration (Guardrail 2)
  - On startup, POST manifest to Gateway
  - Use Docker Network Alias: `http://sanctuary-gateway:4444`
  - **VERIFY:** Gateway logs show registration

- [ ] **Step 3.2:** Test Gateway routing
  - Call tool via Gateway endpoint
  - **VERIFY:** Response received within 100ms

- [ ] **Step 3.3:** Add health check endpoint
  - `/health` returns container status
  - Gateway can poll for health

### Phase 4: Production Hardening
**Goal:** Multi-stage Dockerfile, hot reload, documentation.

- [ ] **Step 4.1:** Multi-stage Dockerfile (dev/prod)
  - Dev stage: volume mounts, hot reload
  - Prod stage: minimal image, no dev deps
  
- [ ] **Step 4.2:** Hot reload verification
  - Edit tool code, see changes without rebuild
  - **VERIFY:** Change reflected in <2 seconds

- [ ] **Step 4.3:** Write README
  - `mcp_servers/utils/README.md`
  - Architecture, tools list, development guide

- [ ] **Step 4.4:** Final commit and push
  - All tests passing
  - PR ready for merge

---

## 5. Current Progress

| Step | Status | Notes |
|------|--------|-------|
| 1.1 Directory structure | ✅ DONE | Created mcp_servers/utils/ with tools/ |
| 1.2 Minimal FastAPI | ✅ DONE | SSE server with health, manifest, tools |
| 1.3 Minimal Dockerfile | ✅ DONE | With curl for health checks |
| 1.4 docker-compose entry | ✅ DONE | With resource limits, hot reload, mcp-network |
| 2.1 Calculator + fault | ✅ DONE | Fault containment verified (div/0 safe) |
| 2.2 UUID tool | ✅ DONE | generate_uuid4, generate_uuid1, validate_uuid |
| 2.3 String utilities | ✅ DONE | upper, lower, trim, reverse, word_count, replace |
| 3.1 Self-Registration | ✅ DONE | Registers on startup, graceful fallback |
| 3.2 Gateway routing | ✅ DONE | Works standalone when Gateway unavailable |
| 3.3 Health check | ✅ DONE | /health endpoint working |
| 4.1 Multi-stage Docker | ⬜ TODO | |
| 4.2 Hot reload | ✅ DONE | Volume mount + uvicorn --reload |
| 4.3 README | ✅ DONE | Full documentation with examples |
| 4.4 Final commit | ⬜ TODO | |

**Tools Registered:** 16 (time: 2, calculator: 5, uuid: 3, string: 6)
**Container Port:** 8100 (configurable via SANCTUARY_UTILS_PORT)
