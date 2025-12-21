# Agent Persona MCP Server Documentation

## Overview

Agent Persona MCP manages individual LLM agent execution, persona injection, and per-agent session state. It provides the low-level interface for executing tasks with specific agent roles (auditor, strategist, coordinator).

## Key Concepts

- **Persona Management:** Load and manage agent personas (system prompts)
- **Session State:** Maintain conversation history per agent role
- **LLM Interface:** Abstraction over multiple LLM providers (Ollama, OpenAI, Gemini)
- **Extensibility:** Support for custom personas

## Architecture

```
External LLM → Agent Persona MCP (Server)
                    ↓
            LLM Engines (Ollama/OpenAI/Gemini)
            Persona Files (.txt)
            Session State (.json)
```

## Server Implementation

- **Server Code:** [mcp_servers/agent_persona/server.py](../../../mcp_servers/agent_persona/server.py)
- **Operations:** [mcp_servers/agent_persona/agent_persona_ops.py](../../../mcp_servers/agent_persona/agent_persona_ops.py)
- **LLM Client:** [mcp_servers/agent_persona/llm_client.py](../../../mcp_servers/agent_persona/llm_client.py)

## Testing

- **Test Suite:** [tests/mcp_servers/agent_persona/](../../../tests/mcp_servers/agent_persona/)
- **Status:** ✅ 34/34 tests passing
- **Coverage:** 80%+ (comprehensive suite with edge cases)

## Operations

### `persona_dispatch`
Dispatch a task to a specific persona agent

**Example:**
```python
persona_dispatch(
    role="auditor",
    task="Review the test coverage for the Git MCP server",
    maintain_state=True,
    engine="ollama",
    model_name="Sanctuary-Qwen2-7B:latest"
)
```

### `persona_list_roles`
List all available persona roles (built-in + custom)

### `persona_get_state`
Get conversation state for a specific role

### `persona_reset_state`
Reset conversation state for a specific role

### `persona_create_custom`
Create a new custom persona

## Built-in Personas

- **Coordinator:** Task planning and execution oversight
- **Strategist:** Long-term planning and risk assessment
- **Auditor:** Quality assurance and compliance verification

## Performance

- **Latency:** 30-60 seconds per agent call (LLM inference)
- **Model:** Sanctuary-Qwen2-7B:latest via Ollama
- **Session Management:** Persistent state across calls

## Related ADRs

- [ADR 040: Agent Persona MCP Architecture](../../../ADRs/040_agent_persona_mcp_architecture__modular_council_members.md)
- [ADR 042: Separation of Council MCP and Agent Persona MCP](../../../ADRs/042_separation_of_council_mcp_and_agent_persona_mcp.md)

## Status

✅ **Fully Operational** - Tested with Sanctuary-Qwen2-7B model
