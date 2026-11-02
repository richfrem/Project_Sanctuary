# Separation of Council MCP and Agent Persona MCP

**Status:** proposed
**Date:** 2025-12-02
**Author:** AI Assistant (Antigravity)


---

## Context

Project Sanctuary implements a 12-domain MCP architecture following Domain-Driven Design (DDD) principles. Two of these domains are:

1. **Agent Persona MCP** - Manages individual LLM agent execution, persona injection, and per-agent session state
2. **Council MCP** - Orchestrates multi-agent deliberation workflows, managing the sequence and state machine of multi-step processes

During architecture review, the question arose: Should these be merged into a single "Orchestrator-Council MCP" to reduce complexity?

**Key Considerations:**
- The Council MCP acts as a **flow control and state machine** (high-level logic, stable, easy to test)
- The Agent Persona MCP acts as an **LLM interface layer** (low-level inference, high latency, volatile)
- LLM calls via Agent Persona are the slowest part of the process (30-60 seconds per agent on self-hosted models)
- The Council MCP is a **client** to the Agent Persona MCP, calling it multiple times per deliberation round

## Decision

**We will maintain Council MCP and Agent Persona MCP as separate, independent MCP servers.**

**Rationale:**

1. **Single Responsibility Principle (SRP):** Each MCP has a distinct bounded context:
   - Agent Persona: LLM execution and persona management
   - Council: Workflow orchestration and multi-agent coordination

2. **Scalability:** Keeping them separate allows:
   - Independent scaling of the LLM interface layer (Agent Persona)
   - Potential parallelization of agent calls in future iterations
   - Resource allocation based on actual bottlenecks

3. **Testability:** Separation enables:
   - Unit testing of Agent Persona in isolation (verify LLM integration works)
   - Integration testing of Council (verify orchestration logic works)
   - Clear test boundaries and responsibilities

4. **Maintainability:** Changes to:
   - Persona definitions (system prompts, roles) only affect Agent Persona MCP
   - Orchestration logic (deliberation rounds, state machine) only affect Council MCP
   - Reduces coupling and blast radius of changes

5. **DDD Compliance:** Maintains the 12-domain architecture's integrity and prevents domain bleed

**Implementation:**
- Council MCP exposes `council_dispatch()` and `council_list_agents()` tools
- Agent Persona MCP exposes `persona_dispatch()`, `persona_list_roles()`, etc.
- Council MCP acts as a **client** to Agent Persona MCP, calling it internally
- Both MCPs remain independently deployable and testable

## Consequences

**Positive:**
- **Modularity:** Each MCP has a single, well-defined responsibility
- **Scalability:** Agent Persona MCP can be scaled independently (LLM calls are the bottleneck)
- **Testability:** Each MCP can be tested in isolation before integration testing
- **Maintainability:** Changes to personas don't require redeploying the orchestration logic
- **Safety:** Separation reduces blast radius of failures in either component

**Negative:**
- **Complexity:** Requires managing two separate MCP servers instead of one
- **Network Overhead:** Additional MCP protocol overhead for inter-server communication
- **Debugging:** Multi-server debugging is more complex than single-server

**Risks:**
- If the Agent Persona MCP is unavailable, the Council MCP cannot function
- Network latency between MCPs could impact performance (mitigated by local deployment)
