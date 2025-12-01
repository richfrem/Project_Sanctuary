# MCP Testing Standards

**Protocol 115: The Tactical Mandate - Documentation as Requirement**

## Overview

Reliability is paramount for the Sanctuary's Nervous System. This document establishes the standard workflow for testing and verifying MCP servers.

## The Testing Workflow

All MCP servers must adhere to this 4-layer testing pyramid:

### 1. Script/Unit Testing (The Foundation) 🧪
**Goal:** Verify the underlying logic *before* it is wrapped in an MCP tool.
**Method:** Pytest unit tests.
**Location:** `tests/mcp_servers/<server_name>/` or `tests/test_<domain>_operations.py`

```bash
# Example
pytest tests/mcp_servers/cortex/test_operations.py -v
```

### 2. Integration Testing 🔗
**Goal:** Verify interactions between components (e.g., Database, Git, Filesystem).
**Method:** Pytest integration tests.
**Location:** `tests/integration/`

```bash
# Example
pytest tests/integration/test_forge_integration.py -v
```

### 3. MCP Tool Verification 🤖
**Goal:** Verify the MCP tool interface (arguments, returns, error handling) works as expected when called by an LLM.
**Method:** Manual verification via Claude Desktop or Antigravity, or automated tool tests.

**Verification Prompt Template:**
> "Please [perform action] using the [tool_name] tool to verify its functionality."

### 4. End-to-End Orchestration 🎼
**Goal:** Verify complex workflows involving multiple MCPs.
**Method:** Council Orchestrator missions.

## Documentation Requirements

Every MCP Server README must include a **Testing** section with:

1.  **Command:** Exact command to run unit tests.
2.  **Results:** A snapshot of passing test output (or link to CI logs).
3.  **Verification:** Instructions for manual verification.

## Test Data Management

- Use `tests/fixtures/` for static test data.
- Clean up any artifacts created during testing (use `tmp_path` fixture in pytest).
- **NEVER** commit test artifacts to the main repository (use `.gitignore`).

## Continuous Integration

(Future Phase)
- All tests must pass before merging to `main`.
- Protocol 101 v3.0 enforces this for Git operations.
