# Agent Persona MCP Architecture - Modular Council Members

**Status:** proposed
**Date:** 2025-11-30
**Author:** Antigravity & User (Council Architecture Evolution)


---

## Context

The Council Orchestrator currently manages three agent personas (Coordinator, Strategist, Auditor) as internal Python objects within a monolithic architecture. Each agent is initialized with a persona seed file and maintains conversation state.

As the MCP ecosystem evolved, we recognized an opportunity to apply the same "separation of concerns" principle that we used for the Council MCP itself. Instead of having agents tightly coupled to the orchestrator, we could extract them as independent MCP servers.

The `PersonaAgent` class in `council_orchestrator/orchestrator/council/agent.py` is already well-designed for this extraction - it loads persona configuration from files and can assume any role. This makes it ideal for a single, parameterized "Agent Persona MCP" server that can instantiate any council member role.

**Current Architecture (v1.0):**
```
Council Orchestrator (Monolithic)
├── Coordinator (Python class)
├── Strategist (Python class)
└── Auditor (Python class)
```

**Proposed Architecture (v2.0):**
```
Council Orchestrator (MCP Client)
├── Calls → Agent Persona MCP (role=coordinator)
├── Calls → Agent Persona MCP (role=strategist)
└── Calls → Agent Persona MCP (role=auditor)
```

The orchestrator would transition from managing agents internally to coordinating them via MCP protocol.

## Decision

We will evolve the Council architecture to use **Agent Persona MCP servers** instead of internal agent objects.

**Implementation:**

1. **Create Agent Persona MCP Server** (`mcp_servers/agent_persona/`)
   - Single MCP server that can assume any persona role
   - Tools: `persona_dispatch(role, task, context)`, `persona_list_roles()`, `persona_get_state(role)`
   - Reuses existing `PersonaAgent` class from `council_orchestrator/orchestrator/council/`

2. **Refactor Council Orchestrator as MCP Client**
   - Orchestrator becomes a coordinator that calls Agent Persona MCP
   - Maintains deliberation logic and consensus mechanism
   - Calls other MCPs (Code, Git, Cortex) for autonomous workflows

3. **Phased Migration Path:**
   - **Phase 1 (Current)**: Monolithic orchestrator with internal agents ✅
   - **Phase 2**: Create Agent Persona MCP, test with one agent (Auditor)
   - **Phase 3**: Migrate all agents to MCP, support dual mode
   - **Phase 4**: Deprecate internal agent implementation, pure MCP client

4. **Benefits:**
   - Each agent independently deployable
   - External agents can consult individual council members
   - Horizontal scaling (multiple agent instances)
   - Polyglot implementation (agents in different languages)

**Naming:** "Agent Persona MCP" (not "Persona MCP" or "Council Member MCP") to emphasize both the agent nature and the configurable persona aspect.

**Design Principle:** Apply separation of concerns at the agent level, just as we did at the MCP server level.

## Consequences

**Positive:**
- **True Modularity**: Each agent is independently deployable and upgradeable
- **Scalability**: Agents can run on different machines, horizontal scaling possible
- **Specialization**: Each agent MCP can have its own specialized tools and capabilities
- **Composability**: External agents can call individual council members directly without full deliberation
- **Polyglot**: Agents can be implemented in different languages
- **Testing**: Individual agents can be tested in isolation
- **Agent Marketplace**: Could swap in different implementations (e.g., different Auditor strategies)

**Negative:**
- **Increased Complexity**: More moving parts (multiple MCP servers vs monolithic)
- **Network Overhead**: Inter-MCP communication adds latency
- **Configuration Complexity**: Need to manage multiple MCP server configurations
- **Debugging Difficulty**: Distributed system debugging is harder than monolithic

**Risks:**
- Agent discovery and service registry complexity
- Consensus mechanism design (how to resolve disagreements)
- Error handling across MCP boundaries
- State management (stateless vs stateful agents)

**Mitigation:**
- **Phased Migration**: Extract one agent at a time (start with Auditor)
- **Dual Mode**: Support both internal and external agents during transition
- **Clear Interfaces**: Define standard agent MCP tool signatures
- **Comprehensive Testing**: Test both individual agents and orchestration
- **Documentation**: Clear migration guide and architecture diagrams
