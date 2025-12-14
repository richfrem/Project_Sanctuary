# Orchestrator MCP Tests

Tests for the Orchestrator MCP, focusing on logic coordination and the critical Strategic Crucible Loop.

## Structure

### 1. Unit/Integration Logic (`integration/`)
**File:** `test_operations.py`
- Tests the `OrchestratorOperations` logic in isolation.
- Mocks out `CortexOperations` to verify flow control without side effects.
- Covers `dispatch_mission` and `run_strategic_cycle` logic.

### 2. End-to-End System Tests (`e2e/`)
**File:** `test_strategic_crucible.py`
- **Protocol 056 Compliance Test.**
- Executes the full "Self-Evolving Loop" with **REAL** (but isolated) dependencies.
- Uses `BaseIntegrationTest` to ensure ChromaDB is available.
- Creates a temporary project structure and verifies:
  1. Research Report Ingestion (Cortex)
  2. Adaptation Packet Generation (Simulated)
  3. Guardian Cache Wakeup (Cortex)

## Running Tests

```bash
# Run logic tests (Fast)
pytest tests/mcp_servers/orchestrator/integration/ -v

# Run E2E System Tests (Starts ChromaDB)
pytest tests/mcp_servers/orchestrator/e2e/ -v
```
