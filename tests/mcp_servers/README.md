# Model Context Protocol (MCP) Server Test Pyramid

This directory contains the **Component Tests** for all individual MCP server implementations (e.g., `rag_cortex`, `council`, `git`). This is the foundation of our entire Test Pyramid, ensuring that each server's internal logic and API contract are robust before high-level orchestration is attempted.

We adhere to the principle of **Designing for Successor-State** (Chronicle Entry 308): A new agent must be able to run and understand the test suite instantly.


## Structure of the Test Pyramid

Every MCP server sub-directory (`<mcp>/`) is structured into three layers.

| Sub-Folder | Scope | Goal | Execution Command (Example) |
| :--- | :--- | :--- | :--- |
| **`unit/`** | Internal functions/classes | Verify atomic logic (e.g., a data validator or utility function) in isolation, **no network calls**. | `pytest tests/mcp_servers/<mcp>/unit/` |
| **`integration/`** | Server-to-DB/Local-API | Verify the server's core operations with its local dependencies (e.g., `rag_cortex` talking to ChromaDB, `git` talking to the local file system). **Minimal mocking.** | `pytest tests/mcp_servers/<mcp>/integration/` |
| **`e2e/`** | MCP Client Call | Verify the full lifecycle of an MCP call as executed by the canonical MCP Client (e.g., Claude Desktop, Antigravity). **Requires all 12 servers to be running.** | `pytest tests/mcp_servers/<mcp>/e2e/` |


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
