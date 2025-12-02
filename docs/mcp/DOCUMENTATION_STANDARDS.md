# MCP Documentation Standards

**Version:** 1.0  
**Status:** Active  
**Effective Date:** 2025-12-01

## 1. Overview

This document establishes the documentation standards for the Project Sanctuary MCP (Model Context Protocol) ecosystem. With 11 distributed MCP servers, consistent and high-quality documentation is essential for maintainability, discoverability, and developer experience.

## 2. Documentation Architecture

The documentation is organized into three layers:

1.  **Central Hub (`docs/mcp/`)**: High-level architecture, standards, and cross-server guides.
2.  **Server Documentation (`mcp_servers/*/README.md`)**: Specific instructions for each MCP server.
3.  **Code Documentation (Docstrings)**: Inline documentation for tools and internal logic.

### 2.1 The Single Source of Truth

*   **Operations Inventory:** `docs/mcp/mcp_operations_inventory.md` is the **Single Source of Truth** for tool availability and testing status.
*   **Architecture:** `docs/mcp/architecture.md` is the authoritative source for system topology.

## 3. MCP Server README Standards

Every MCP server MUST have a `README.md` in its root directory (e.g., `mcp_servers/council/README.md`).

### Required Sections

1.  **Title & Description**: Clear name and purpose of the server.
2.  **Tools**: A table listing available tools, their description, and input arguments.
3.  **Resources**: A table listing available resources (if any).
4.  **Prompts**: A table listing available prompts (if any).
5.  **Configuration**: Required environment variables and `mcp_config.json` snippet.
6.  **Testing**:
    *   Command to run unit tests.
    *   Command to run integration tests.
    *   Verification steps for manual testing.
7.  **Dependencies**: Key libraries used.

**Template:** See `docs/mcp/templates/mcp_server_readme.md`

## 4. Docstring Standards

All Python code MUST use **Google Style** docstrings.

### 4.1 MCP Tools

Every function decorated as an MCP tool (`@mcp.tool()`) MUST have a docstring that includes:

*   **Summary**: A concise one-line description of what the tool does.
*   **Description**: Detailed explanation of behavior, side effects, and edge cases.
*   **Args**: List of arguments with types and descriptions.
*   **Returns**: Description of the return value and structure.
*   **Example**: A usage example (optional but recommended).

**Template:** See `docs/mcp/templates/mcp_tool_docstring.md`

### 4.2 Internal Functions

*   Public functions/classes: Full docstrings required.
*   Private functions (`_func`): Brief summary required.

## 5. Inventory Maintenance

The `docs/mcp/mcp_operations_inventory.md` file tracks the implementation and testing status of every tool.

**When to Update:**
*   **Adding a Tool:** Add a new row with status `❌` (Untested) or `⚠️` (Partial).
*   **Deprecating a Tool:** Move to "Deprecated" section or mark as `[DEPRECATED]`.
*   **Verifying a Tool:** Update status to `✅` only after:
    1.  Unit tests pass.
    2.  Integration/Manual verification is complete.
    3.  Docstrings are compliant.

## 6. Versioning

*   Documentation updates should be committed alongside code changes.
*   Major architectural changes require updating `docs/mcp/architecture.md`.
