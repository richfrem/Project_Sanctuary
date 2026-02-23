# Council MCP Tests

Tests for the Council MCP, comprising the "Deliberative" layer.

## Structure

### 1. Unit Tests (`unit/`)
- `test_routing.py`: Tests Protocol 116 (Container Network vs Localhost) logic.
- `test_tool_dispatch.py`: Tests internal tool calling logic using mocks.

### 2. Integration Tests (`integration/`)
**File:** `test_operations.py`
- Tests `council_dispatch` and `council_list_agents` against **REAL** dependencies.
- **Dependencies:**
  - **Agent Persona MCP:** Must be importable.
  - **Ollama:** Must be running (test skips if not).
  - **RAG Cortex:** Mocked out (to isolate Council logic from RAG latency).
- Verifies that the Council can successfully instantiate agents, query LLMs, and return structured decisions.

## Prerequisites

```bash
ollama serve
```

## Running Tests

```bash
# Run all Council tests
pytest tests/mcp_servers/council/ -v
```
