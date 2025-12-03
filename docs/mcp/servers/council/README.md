# Council MCP Server Documentation

## Overview

Council MCP is a **specialized orchestrator** for multi-agent deliberation workflows. It orchestrates multiple agent calls across deliberation rounds, manages conversation state, and synthesizes consensus.

## Key Concepts

- **Specialized Orchestrator:** Focused on multi-agent deliberation only
- **Tactical Scope:** Multi-round agent discussions (2-5 minutes)
- **Client to:** Agent Persona MCP, Cortex MCP, Code MCP, Git MCP

## Architecture

```
External LLM → Council MCP (Server)
                    ↓ (Client)
            Agent Persona MCP
            Cortex MCP
```

## Documentation

- **[Council vs Orchestrator](council_vs_orchestrator.md)** - Relationship between Council and Orchestrator MCPs
- **[Orchestration Workflows](orchestration_workflows.md)** - Detailed workflow patterns
- **[Orchestration Validation](mcp_orchestration_validation.md)** - Testing and validation
- **[Simple Orchestration Test](simple_orchestration_test.md)** - Basic test scenarios
- **[Complete Orchestration Test](complete_orchestration_test.md)** - Comprehensive test scenarios
- **[Final Orchestration Test](final_orchestration_test.md)** - End-to-end validation

## Server Implementation

- **Server Code:** [mcp_servers/council/server.py](../../../mcp_servers/council/server.py)
- **Operations:** [mcp_servers/council/council_ops.py](../../../mcp_servers/council/council_ops.py)
- **Packets:** [mcp_servers/council/packets/](../../../mcp_servers/council/packets/)

## Testing

- **Test Suite:** [tests/mcp_servers/council/](../../../tests/mcp_servers/council/)
- **Status:** ✅ 3/3 tests passing

## Operations

### `council_dispatch`
Execute task through multi-agent deliberation

**Example:**
```python
council_dispatch(
    task_description="Review Protocol 101 for compliance issues",
    agent="auditor",
    max_rounds=3
)
```

### `council_list_agents`
List available council agents (coordinator, strategist, auditor)

## Related ADRs

- [ADR 039: MCP Server Separation of Concerns](../../../ADRs/039_mcp_server_separation_of_concerns.md)
- [ADR 040: Agent Persona MCP Architecture](../../../ADRs/040_agent_persona_mcp_architecture__modular_council_members.md)
- [ADR 042: Separation of Council MCP and Agent Persona MCP](../../../ADRs/042_separation_of_council_mcp_and_agent_persona_mcp.md)

## Performance

- **Latency:** 30-60 seconds per agent call
- **Bottleneck:** LLM inference via Agent Persona MCP
- **Scalability:** Can be scaled independently

## Status

✅ **Operational** - Refactored to use Agent Persona MCP and Cortex MCP
