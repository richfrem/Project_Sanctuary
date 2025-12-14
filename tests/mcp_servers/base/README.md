# MCP Server Test Structure - Common Patterns

This document defines the **standard structure and implementation patterns** for all MCP server tests across the Project Sanctuary ecosystem.

## Directory Structure

Every MCP server MUST follow this 3-layer test pyramid structure:

```
tests/mcp_servers/<server>/
├── unit/                    # Layer 1: Isolated unit tests
│   ├── __init__.py
│   └── test_*.py
├── integration/             # Layer 2: Real dependency tests
│   ├── __init__.py
│   └── test_*.py
└── e2e/                     # Layer 3: Full MCP protocol tests
    ├── __init__.py
    └── test_*.py
```

## Layer 1: Unit Tests (`unit/`)

### Purpose
Test atomic logic in complete isolation with no external dependencies.

### Characteristics
- **Speed:** Fast (milliseconds)
- **Dependencies:** None (mocked if needed)
- **Base Class:** None required
- **Scope:** Individual functions, classes, validators, parsers

### Standard Pattern

```python
"""Unit tests for <component>."""

import pytest
from mcp_servers.<server>.<module> import function_to_test


def test_function_name():
    """Test specific behavior of function."""
    # Arrange
    input_data = "test input"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output


def test_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

### When to Use
- Testing validators, parsers, formatters
- Testing utility functions
- Testing data transformations
- Testing business logic without I/O

### When NOT to Use
- Testing database operations → Use integration tests
- Testing API calls → Use integration tests
- Testing file I/O → Use integration tests
- Testing MCP protocol → Use E2E tests

---

## Layer 2: Integration Tests (`integration/`)

### Purpose
Test server operations with **real local dependencies** (ChromaDB, Ollama, Git).

### Characteristics
- **Speed:** Medium (seconds)
- **Dependencies:** ChromaDB (port 8000), Ollama (port 11434), Git repo
- **Base Class:** `BaseIntegrationTest` (enforces dependency checking)
- **Scope:** Server ↔ Local Service interactions

### Standard Pattern

```python
"""Integration tests for <server> with real dependencies."""

import pytest
from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.<server>.operations import Operations


class Test<Server>Live(BaseIntegrationTest):
    """
    Live integration tests for <Server> MCP.
    
    These tests require real services to be running:
    - ChromaDB on port 8000 (if applicable)
    - Ollama on port 11434 (if applicable)
    - Git repository initialized (if applicable)
    """
    
    def get_required_services(self):
        """Define required services for these tests."""
        return [
            ("localhost", 8000, "ChromaDB"),
            ("localhost", 11434, "Ollama")
        ]
    
    def test_operation_with_real_dependency(self):
        """Test operation with real service connectivity."""
        # Arrange
        ops = Operations()
        
        # Act
        result = ops.perform_operation()
        
        # Assert
        assert result["status"] == "success"
```

### Dependency Auto-Checking

The `BaseIntegrationTest` class automatically:
1. Checks if required services are running before each test
2. **Skips** tests if services are missing (local development)
3. **Fails** tests if services are missing (CI environment with `CI=true`)

### When to Use
- Testing database ingestion/queries
- Testing LLM API calls
- Testing Git operations
- Testing file system operations
- Testing any operation that requires real I/O

### When NOT to Use
- Testing pure logic → Use unit tests
- Testing full MCP protocol → Use E2E tests

---

## Layer 3: E2E Tests (`e2e/`)

### Purpose
Test the **full MCP client call lifecycle** through the MCP protocol.

### Characteristics
- **Speed:** Slow (minutes)
- **Dependencies:** All 12 MCP servers running
- **Base Class:** `BaseE2ETest` (provides MCP client utilities)
- **Scope:** Complete user scenarios via MCP protocol

### Standard Pattern

```python
"""E2E tests for <Server> MCP via MCP protocol."""

import pytest
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest


@pytest.mark.e2e
class Test<Server>E2E(BaseE2ETest):
    """
    End-to-end tests for <Server> MCP server via MCP protocol.
    
    These tests verify:
    - Full MCP client → server communication
    - Complete tool call lifecycle
    - Real responses from the MCP server
    """
    
    @pytest.mark.asyncio
    async def test_tool_via_mcp_client(self, mcp_servers):
        """Test <tool_name> through MCP client."""
        # TODO: Implement when MCP client is integrated
        
        # Expected usage:
        # result = await self.call_mcp_tool(
        #     "tool_name",
        #     {"param": "value"}
        # )
        # 
        # self.assert_mcp_success(result)
        # assert result["data"] == expected_value
        
        pytest.skip("MCP client integration pending - structure established")
```

### Infrastructure

E2E tests use the **`mcp_servers` pytest fixture** (defined in `tests/conftest.py`):

```python
@pytest.fixture(scope="session")
def mcp_servers():
    """
    Automatically starts all 12 MCP servers before tests.
    Uses the standard start_mcp_servers.py script.
    Tears down all servers after test session completes.
    """
```

This fixture:
1. Starts all servers using `mcp_servers/start_mcp_servers.py --run`
2. Waits for initialization (5 seconds)
3. Yields control to tests
4. Cleanly shuts down all servers on teardown

### When to Use
- Testing complete MCP tool workflows
- Testing multi-step MCP operations
- Testing MCP protocol compliance
- Testing end-user scenarios

### When NOT to Use
- Testing individual functions → Use unit tests
- Testing database operations → Use integration tests
- Testing before MCP client is integrated → Mark with `pytest.skip()`

---

## Running Tests

### Run All Tests for a Server
```bash
# Run all 3 layers for a specific server
pytest tests/mcp_servers/rag_cortex/ -v
```

### Run Specific Layer
```bash
# Layer 1: Unit tests (fast, no dependencies)
pytest tests/mcp_servers/rag_cortex/unit/ -v

# Layer 2: Integration tests (requires ChromaDB, Ollama)
pytest tests/mcp_servers/rag_cortex/integration/ -v

# Layer 3: E2E tests (requires all 12 MCP servers)
pytest tests/mcp_servers/rag_cortex/e2e/ -v -m e2e
```

### Run All Tests Across All Servers
```bash
# All unit tests (fast)
pytest tests/mcp_servers/*/unit/ -v

# All integration tests (medium)
pytest tests/mcp_servers/*/integration/ -v

# All E2E tests (slow)
pytest tests/mcp_servers/*/e2e/ -v -m e2e
```

---

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit          # Layer 1 (optional, implied by directory)
@pytest.mark.integration   # Layer 2 (optional, implied by directory)
@pytest.mark.e2e           # Layer 3 (REQUIRED for E2E tests)
@pytest.mark.asyncio       # For async test functions
```

---

## Base Classes Reference

### `BaseIntegrationTest`
**Location:** `tests/mcp_servers/base/base_integration_test.py`

**Purpose:** Enforce dependency checking for integration tests

**Required Method:**
```python
def get_required_services(self) -> List[Tuple[str, int, str]]:
    """Return list of (host, port, service_name) tuples."""
    return [("localhost", 8000, "ChromaDB")]
```

### `BaseE2ETest`
**Location:** `tests/mcp_servers/base/base_e2e_test.py`

**Purpose:** Provide MCP client utilities for E2E tests

**Utility Methods:**
```python
async def call_mcp_tool(self, tool_name: str, arguments: Dict) -> Dict:
    """Call an MCP tool through the MCP client."""
    
def assert_mcp_success(self, result: Dict, message: str = ""):
    """Assert that an MCP tool call was successful."""
```

---

## Checklist for New MCP Server Tests

When creating tests for a new MCP server, ensure:

- [ ] Directory structure follows the 3-layer pyramid
- [ ] Each layer has `__init__.py`
- [ ] Unit tests are isolated (no external dependencies)
- [ ] Integration tests inherit from `BaseIntegrationTest`
- [ ] Integration tests define `get_required_services()`
- [ ] E2E tests inherit from `BaseE2ETest`
- [ ] E2E tests use `@pytest.mark.e2e` decorator
- [ ] E2E tests use `mcp_servers` fixture
- [ ] All tests follow naming convention: `test_*.py`
- [ ] Test docstrings clearly describe what is being tested

---

## Related Documentation

- **Test Pyramid Overview:** `tests/mcp_servers/README.md`
- **ADR 047:** Systemic Integration Testing Mandate
- **ADR 054:** Harmonize RAG Cortex Test Structure
- **Server Launcher:** `mcp_servers/start_mcp_servers.py`
