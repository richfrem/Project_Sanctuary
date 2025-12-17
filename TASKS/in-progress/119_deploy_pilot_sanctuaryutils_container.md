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
