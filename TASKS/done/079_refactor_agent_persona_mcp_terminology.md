# TASK: Refactor Agent Persona MCP Terminology

**Status:** done
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Refactor Agent Persona MCP to use standard terminology and decouple from legacy council_orchestrator code

## 2. Deliverables

1. mcp_servers/lib/agent_persona/agent.py
2. mcp_servers/lib/agent_persona/llm_client.py
3. Updated agent_persona_ops.py

## 3. Acceptance Criteria

- Agent Persona MCP no longer imports from council_orchestrator
- Terminology updated: Substrate -> LLMClient, Awakening Seed -> System Prompt
- All tests pass with new implementation
