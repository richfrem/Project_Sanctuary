# Model Context Protocol (MCP) Server Test Pyramid

This directory contains the **Component Tests** for all individual MCP server implementations (e.g., `rag_cortex`, `council`, `git`). This is the foundation of our entire Test Pyramid, ensuring that each server's internal logic and API contract are robust before high-level orchestration is attempted.

We adhere to the principle of **Designing for Successor-State** (Chronicle Entry 308): A new agent must be able to run and understand the test suite instantly.


## Structure of the Test Pyramid

Every MCP server sub-directory (`<mcp>/`) is structured into three layers:

| Layer | Sub-Folder | Scope | Base Class | Dependencies | Speed |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **`unit/`** | Internal functions/classes | None (isolated) | None | Fast (ms) |
| **2** | **`integration/`** | Server ↔ Local Services | `BaseIntegrationTest` | ChromaDB, Ollama, Git | Medium (sec) |
| **3** | **`e2e/`** | Full MCP Protocol | `BaseE2ETest` | All 12 MCP servers | Slow (min) |

### Layer 1: Unit Tests (`unit/`)
- **Focus:** Atomic logic in isolation (validators, parsers, utilities)
- **Dependencies:** None (mocked if needed)
- **Base Class:** None required
- **Run:** `pytest tests/mcp_servers/<mcp>/unit/ -v`

**Example:**
```python
def test_protocol_validator():
    """Test protocol number validation logic."""
    assert validate_protocol_number(101) is True
    assert validate_protocol_number(-1) is False
```

### Layer 2: Integration Tests (`integration/`)
- **Focus:** Server operations with real local dependencies
- **Dependencies:** ChromaDB (port 8000), Ollama (port 11434), Git repo
- **Base Class:** `BaseIntegrationTest` (auto-checks dependencies)
- **Run:** `pytest tests/mcp_servers/<mcp>/integration/ -v`

**Example:**
```python
from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest

class TestRAGCortexLive(BaseIntegrationTest):
    def get_required_services(self):
        return [("localhost", 8000, "ChromaDB")]
    
    def test_ingest_and_query(self):
        # Tests real ChromaDB connectivity
        ...
```

### Layer 3: E2E Tests (`e2e/`)
- **Focus:** Full MCP client call lifecycle via MCP protocol
- **Dependencies:** All 12 MCP servers running (via `start_mcp_servers.py`)
- **Base Class:** `BaseE2ETest` (provides MCP client utilities)
- **Run:** `pytest tests/mcp_servers/<mcp>/e2e/ -v`

**Infrastructure:**
E2E tests use the `mcp_servers` pytest fixture (defined in `tests/conftest.py`) 
which automatically starts all servers before tests and tears them down after.

**Example:**
```python
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

@pytest.mark.e2e
class TestRAGCortexE2E(BaseE2ETest):
    @pytest.mark.asyncio
    async def test_query_via_mcp_client(self, mcp_servers):
        result = await self.call_mcp_tool(
            "cortex_query",
            {"query": "What is Protocol 101?", "max_results": 3}
        )
        self.assert_mcp_success(result)
```


## Execution Quick Reference

### Run All Component Tests

Use the following command to execute all unit and component-level integration tests within this directory:

```bash
# Runs unit and internal integration tests for all 12 MCP servers
pytest tests/mcp_servers/
```

### Run a Specific Server's Tests

```bash
# Example: Run all tests for the RAG Cortex MCP
pytest tests/mcp_servers/rag_cortex/

# Example: Run only the unit tests for the Council MCP
pytest tests/mcp_servers/council/unit/
```


## Related Test Suites

For a complete picture of system health, you must also run the higher-level test suites:

- **Full Integration Suite**
  - Location: `tests/integration/`
  - Purpose: Validates multi-step, multi-MCP workflows (e.g., Auditor -> Strategist chaining, Strategic Crucible Loop Protocol 056).
  - Command: `pytest tests/integration/`

- **System Health Checks**
  - Location: `tests/system/` and `tests/verify_wslenv_setup.py`
  - Purpose: Ensures the environment and the multi-server configuration files are correctly deployed.
  - Command: `pytest tests/system/`


---

If you'd like, I can:

- add a short CI snippet (GitHub Actions) that runs unit and integration layers on PRs, or
- add a validator script that verifies `tests/mcp_servers/` follows the expected folder layout for each MCP, or
- run a quick local check to ensure the README's code blocks contain valid JSON where applicable.

(See repository attachments for referenced files and further context.)
