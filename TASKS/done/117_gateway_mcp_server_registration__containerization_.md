# TASK: Gateway MCP Server Registration & Containerization Strategy

**Status:** complete
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Task 116 (External Gateway Integration - Basic Connectivity)
**Related Documents:** ADR 058 (Gateway Decoupling), Task 116, docker-compose.yml, mcp_servers/ directory

---

## 1. Objective

Plan and execute the next phase of gateway integration: registering Project Sanctuary's MCP servers with the external gateway and determining the containerization strategy for the 10 script-based servers.

## 2. Deliverables

1. Permission audit document comparing current vs required API token scopes
2. Containerization strategy ADR (which servers to containerize and why)
3. Dockerfile templates for script-based MCP servers
4. Gateway server registration script/process
5. E2E test demonstrating tool call through gateway to containerized MCP server

## 3. Acceptance Criteria

- Gateway can successfully register and route to at least 3 MCP servers (2 existing containers + 1 newly containerized)
- API token permissions verified for server registration operations
- Clear decision documented on containerization strategy (all 12 vs selective)
- Dockerfile templates created for script-based MCP servers
- Test suite validates end-to-end tool calls through gateway

## Notes

**Current State:**
- 2/12 MCP servers are containerized (ollama-model-mcp, vector-db/ChromaDB)
- 10/12 MCP servers run via Python scripts
- Gateway is accessible at localhost:4444 with API token auth
- Basic connectivity tests passing (pulse, circuit breaker, handshake)
**Key Questions to Answer:**
1. Does current API token have permissions to register MCP servers with gateway?
2. Should all 10 remaining servers be containerized, or only specific ones?
3. What's the registration process for MCP servers with the gateway?
4. How does the gateway route tool calls to registered servers?
**Research Needed:**
- Review gateway API documentation for server registration endpoints
- Check API token scopes/permissions in gateway admin
- Analyze which MCP servers benefit most from containerization
- Understand gateway's server discovery/routing mechanism

**Status Change (2025-12-17):** backlog → complete
Strategy defined in ADR 060. Implementation broken out into Task 119 (Pilot) and subsequent fleet tasks.
