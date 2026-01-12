# Learning MCP Tests (Protocol 128)

Server-specific tests for the Learning MCP, verifying the Recursive Learning Loop (Scout, Seal, Chronicle).

## Structure

### 1. Integration Tests (`integration/`)
**File:** [test_operations.py](integration/test_operations.py)
- **Primary Test Suite.**
- Validates all Learning tools (`learning_debrief`, `capture_snapshot`, `persist_soul`, `guardian_wakeup`) using mocks for Git and filesystem state.
- Follows the single-file integration pattern.

### 2. E2E Tests (`e2e/`)
**File:** [test_operations_e2e.py](e2e/test_operations_e2e.py)
- Verifies live tool connectivity for the recursive learning loop.
- Ensures the server correctly executes Phase I (Scout), Phase V (Seal), and Phase VI (Chronicle) of the continuity protocol.
- Confirms the Guardian Wakeup digestive process functions in a live server environment.

## Logic Overview
The Learning MCP is decoupled from the Vector Database. It focuses on repository state, git metadata, and session-level soul persistence. Consequently, these tests do not require ChromaDB or Ollama. They rely on the local filesystem and (optionally) HuggingFace API access for soul persistence.

```bash
# Run all Learning tests (Integration + E2E)
pytest tests/mcp_servers/learning/ -v
```
