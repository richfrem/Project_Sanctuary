# TASK: Council MCP Polymorphic Model Refactoring

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** Requires #093 (Containerize Ollama Model Service) - The ollama_model_mcp network service must be running before Council can route to it
**Related Documents:** Protocol 108 (Federated Deployment), Protocol 115 (Task Protocol), Task #093

---

## 1. Objective

Restore and enforce Polymorphic Model Routing within the Council MCP. Ensure that the Council respects the model preference passed down by the Orchestrator (Gemini, GPT, or Ollama) during multi-agent deliberation, fully integrating the Forge LLM MCP's capabilities (Protocol P108).

## 2. Deliverables

1. Refactored Council MCP methods to accept model_preference parameter
2. Updated council_dispatch and create_cognitive_task methods with polymorphic routing logic
3. Test suite verifying GEMINI and OLLAMA model preference routing
4. Updated API documentation reflecting new model_preference parameter

## 3. Acceptance Criteria

- All Council MCP methods that call the Forge LLM MCP accept model_preference: Optional[str]
- Test verifies model_preference='GEMINI' routes to Gemini API
- Test verifies model_preference='OLLAMA' routes to containerized ollama_model_mcp service
- Documentation updated with model_preference parameter specification
