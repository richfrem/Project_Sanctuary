## E2E (Headless)

This folder contains headless end-to-end tests that exercise ADR tool routing.

Run headless ADR tests:

pytest tests/mcp_servers/adr/e2e -m headless -q

Note: These tests use `MCPClient.route_query` to simulate headless client routing. They are intended for CI/nightly runs and do not make servers available to the IDE/Copilot.
# ADR Server Test Suite

## Overview
This directory contains the test suite for the `mcp_servers.adr` module, ensuring compliance with **Protocol 122** (Schema Validation) and sequential integrity rules.

## Structure

```
tests/mcp_servers/adr/
├── conftest.py              # Shared fixtures (Temp ADR directory)
├── e2e/                     # End-to-End Tests
│   └── test_adr_e2e.py      # Full lifecycle verification
├── integration/             # Integration Tests
│   └── test_operations.py   # ADROperations class integration
├── unit/                    # Unit Tests
│   └── test_validator.py    # Validator logic checks
└── test_adr_connectivity.py # Basic server connectivity check
```

## Running Tests

Run the full suite from the project root:

```bash
pytest tests/mcp_servers/adr/ -v
```

### Specific Categories

**Integration Tests:**
```bash
pytest tests/mcp_servers/adr/integration/ -v
```

**Unit Tests:**
```bash
pytest tests/mcp_servers/adr/unit/ -v
```

## Key Test Cases

### Lifecycle (`e2e/test_adr_e2e.py`)
- **Creation:** Verifies auto-numbering (001, 002...).
- **Status Updates:** Checks valid transitions (Proposed -> Accepted).
- **Search:** Verifies context indexing.

### Integrity
- **Immutability:** Ensures ADRs cannot be deleted via standard tools.
- **Numbering:** Ensures no gaps or duplicates in ADR sequence.

## Requirements
- `pytest`
- `fastmcp`
