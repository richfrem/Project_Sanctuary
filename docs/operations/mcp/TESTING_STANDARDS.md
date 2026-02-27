# Agent Plugin Integration Testing Standards
**(Based on ADR 053 & Protocol 115)**

## 1. The 3-Layer Test Pyramid (ADR 053)

All Agent Plugin Integration servers must adhere to the standardized 3-layer test pyramid.

### Layer 1: Unit Tests (`unit/`)
- **Purpose:** Test atomic logic in complete isolation (functions, classes).
- **Dependencies:** None (mock all external interactions).
- **Speed:** Fast (< 10ms).
- **Location:** `tests/mcp_servers/<server>/unit/`.

### Layer 2: Integration Tests (`integration/`)
- **Purpose:** Test server operations with *real local dependencies*.
- **Dependencies:** 
    - Real Vector DB (Port 8000).
    - Real Ollama (Port 11434).
    - Real Filesystem (via tmp_path).
- **Base Class:** Must inherit from `BaseIntegrationTest`.
- **Speed:** Medium (Seconds).
- **Location:** `tests/mcp_servers/<server>/integration/`.

### Layer 3: End-to-End Tests (`e2e/`)
- **Purpose:** Test full Agent Plugin Integration protocol lifecycle (Client -> Server -> Tool -> Result).
- **Dependencies:** All 15 Agent Plugin Integration servers running via `start_mcp_servers.py`.
- **Base Class:** Must inherit from `BaseE2ETest`.
- **Speed:** Slow (Minutes).
- **Location:** `tests/mcp_servers/<server>/e2e/`.

## 2. Test Execution

### Running Tests
```bash
# Run all tests (fast)
pytest

# Run including integration (requires services up)
pytest tests/mcp_servers/cortex/integration/

# Run slow E2E tests
pytest -m e2e
```

## 3. Documentation Requirements

Every Agent Plugin Integration Server `README.md` must include a **Testing** section answering:
1.  **Unit:** How to run unit tests?
2.  **Integration:** What services must be running?
3.  **Verification:** How to manually verify via Claude Desktop?
