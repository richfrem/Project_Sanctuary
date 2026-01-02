# TASK: Implement Protocol 119 - Multi-Model Abstraction Layer

**Status:** superseded
**Priority:** Medium
**Lead:** Claude (AI Research)
**Dependencies:** None
**Related Documents:** docs/mcp/analysis/microsoft_agent_analysis.md
**Superseded By:** Task #094 (Council MCP Polymorphic Model Refactoring)

---

## Supersession Note

This task has been **superseded by T094** which implemented the multi-model abstraction layer as part of the Council MCP polymorphic routing refactoring.

### What T094 Delivered (Superseding T042)

1. ✅ **Model Abstraction Layer:** Already existed (`LLMClient` abstract base class in `agent_persona/llm_client.py`)
2. ✅ **Provider Adapters:** `OllamaClient`, `OpenAIClient` implementations
3. ✅ **Model Router:** `model_preference` parameter enables polymorphic routing:
   - `model_preference='OLLAMA'` → Routes to `http://ollama_model_mcp:11434` (Protocol 116)
   - `model_preference='GEMINI'` → Routes to Gemini API
   - `model_preference='GPT'` → Routes to OpenAI API

### Key Differences from Original Scope

- T094 focused on **Protocol 116 compliance** (container network isolation) as the primary driver
- Implemented polymorphic routing via `model_preference` parameter rather than task complexity-based routing
- Integrated with existing Agent Persona MCP architecture rather than creating new abstraction

---

## Original Objective

Abstract the LLM interface to support multiple providers (Claude, GPT, Gemini, local models) and enable model routing based on task complexity, reducing vendor lock-in and optimizing costs.

## Original Deliverables

1.  **Model Abstraction Layer:** A unified interface for interacting with different LLM providers.
2.  **Provider Adapters:** Implementations for Anthropic, OpenAI, Google, and generic local models (e.g., via Ollama/Llama.cpp).
3.  **Model Router:** Logic to select the appropriate model based on task requirements/configuration.

## Original Acceptance Criteria

-   [x] Define a generic `ModelProvider` interface. (Delivered as `LLMClient`)
-   [x] Implement adapters for at least 2 providers. (Delivered: Ollama, OpenAI)
-   [x] Update existing agents to use the abstraction layer. (Delivered via Council MCP)
-   [x] Verify that switching models works via configuration. (Delivered via `model_preference`)

---

## References

- [T094: Council MCP Polymorphic Model Refactoring](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/tasks/done/094_council_mcp_polymorphic_model_refactoring.md)
- [Protocol 116: Container Network Isolation](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/01_PROTOCOLS/116_Container_Network_Isolation.md)
