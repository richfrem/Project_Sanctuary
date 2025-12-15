# Adoption of Dynamic MCP Gateway Pattern

**Status:** proposed
**Date:** 2025-12-15
**Author:** Antigravity (Research: Task 110)


---

## Context

Project Sanctuary currently loads 12+ MCP servers statically in Claude Desktop configuration. Each server injects its complete tool definitions into the LLM context window on startup, consuming approximately 8,000 tokens. This creates several critical problems:

1. **Context Window Pollution:** Tool definitions occupy valuable context space that could be used for actual work
2. **Rigid Configuration:** Adding new servers requires manual configuration file updates
3. **Scalability Ceiling:** Cannot exceed ~20 servers before context saturation
4. **Inefficient Resource Usage:** All servers loaded regardless of whether they're needed for current task

Chronicle 308 (Doctrine of Successor State) mandates context efficiency as a strategic priority. The current static loading model is fundamentally incompatible with scaling to 100+ tools while maintaining cognitive clarity.

Research (Task 110) validated that the "MCP Gateway Pattern" with late binding is a proven architectural solution used in production by Skywork.ai, Gravitee, and others. The pattern aligns with Anthropic's official guidance on routing patterns for AI agents.

## Decision

We will implement a **Dynamic MCP Gateway** (codename: "Sanctuary Broker") as the single MCP server registered in Claude Desktop. This Gateway will:

1. **Act as Unified Entry Point:** Replace 12+ static server registrations with ONE broker endpoint
2. **Implement Late Binding:** Dynamically discover and route tool requests to capability-specific backend servers at runtime
3. **Use Service Registry:** Maintain SQLite-based registry of available MCP servers and their capabilities
4. **Provide Tool Summaries:** Inject lightweight summaries into context, loading full definitions only when needed
5. **Enforce Security Allowlists:** Centralized validation of permitted tools and operations via project_mcp.json manifests
6. **Enable Just-in-Time Loading:** Backend servers started/stopped based on actual usage patterns

**Architecture:**
- Gateway: Python FastMCP server in Podman container (sanctuary-broker-mcp)
- Registry: SQLite database mapping tools to backend servers
- Router: Intent classifier directing requests to appropriate servers
- Proxy: JSON-RPC 2.0 message forwarder to backend MCP servers
- Allowlist: Security layer preventing unauthorized tool execution

**Configuration Model:**
Projects define required capabilities in project_mcp.json. Gateway reads manifest and enables only specified servers, reducing context overhead by 94% (500 tokens vs 8,000 tokens).

## Consequences

**Positive Consequences:**
1. **Massive Context Efficiency:** 94% reduction in tool definition overhead (8,000 → 500 tokens)
2. **Unlimited Scalability:** Can support 100+ tools without context saturation
3. **Dynamic Capability Discovery:** New servers auto-discovered without configuration changes
4. **Centralized Security:** Single chokepoint for authentication, authorization, and audit logging
5. **Flexible Deployment:** Backend servers can be updated, scaled, or replaced independently
6. **Project-Specific Tool Sets:** Different projects can enable different capabilities via manifests
7. **Proven Pattern:** Production-validated architecture reduces implementation risk

**Negative Consequences:**
1. **Single Point of Failure:** Gateway outage blocks all MCP functionality (Mitigation: health checks, auto-restart, fallback to direct access)
2. **Additional Latency:** Extra hop through Gateway adds 10-50ms per request (Acceptable for human-in-loop workflows)
3. **Increased Complexity:** More moving parts to maintain and debug (Mitigation: comprehensive testing, monitoring, documentation)
4. **Migration Effort:** Requires refactoring Claude Desktop configuration and testing all 12 servers through Gateway
5. **Routing Logic Risk:** Incorrect routing could send requests to wrong server (Mitigation: schema validation, extensive testing)

**Security Risks:**
1. **Allowlist Bypass:** Malicious prompts could attempt to invoke unauthorized tools (Mitigation: multi-layer validation, audit logging)
2. **Gateway Compromise:** If Gateway is compromised, attacker gains access to all backend servers (Mitigation: container isolation, principle of least privilege, monitoring)

**Operational Impact:**
- Requires new monitoring for Gateway health and routing metrics
- Adds Gateway container to Podman orchestration stack
- Necessitates updated documentation and runbooks

**Overall Assessment:** Benefits significantly outweigh risks. This is the correct architectural evolution for Sanctuary's scale and ambitions.
