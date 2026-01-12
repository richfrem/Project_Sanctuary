# MCP Documentation Standards

**Version:** 1.0  
**Status:** Active  
**Effective Date:** 2025-12-01

## 1. Overview

This document establishes the documentation standards for the Project Sanctuary MCP (Model Context Protocol) ecosystem. With 11 distributed MCP servers, consistent and high-quality documentation is essential for maintainability, discoverability, and developer experience.

## 2. Documentation Architecture

The documentation is organized into three layers:

1.  **Central Hub (`docs/architecture/mcp/`)**: High-level architecture, standards, and cross-server guides.
2.  **Server Documentation (`mcp_servers/*/README.md`)**: Specific instructions for each MCP server.
3.  **Code Documentation (Docstrings)**: Inline documentation for tools and internal logic.

### 2.1 The Single Source of Truth

*   **Operations Inventory:** `docs/architecture/mcp/mcp_operations_inventory.md` is the **Single Source of Truth** for tool availability and testing status.
*   **Architecture:** `docs/architecture/mcp/architecture.md` is the authoritative source for system topology.

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

**Template:** See `docs/architecture/mcp/templates/mcp_server_readme.md`

## 4. Docstring Standards & Hybrid Mandate (ADR 075)

All Python code MUST use **Google Style** docstrings AND **ASCII Banners** (The Hybrid Mandate).

### 4.1 The Hybrid Mandate
Code must be readable by both Humans (scrolling) and Machines (indexing).

1.  **File Headers**: Every file must start with:
    ```python
    #============================================
    # path/to/file.py
    # Purpose: Brief description.
    # Role: Architecture Layer.
    #============================================
    ```

2.  **Method Headers** (The Signpost): Immediately above `def`:
    ```python
    #============================================
    # Method: my_function
    # Purpose: What it does.
    # Returns: What it gives back.
    #============================================
    def my_function(): ...
    ```

### 4.2 MCP Tools
Every function decorated as an MCP tool (`@mcp.tool()`) MUST have a docstring that includes:

*   **Summary**: A concise one-line description.
*   **Description**: Detailed explanation.
*   **Args**: List of arguments.
*   **Returns**: Description of return value.

### 4.3 Internal Functions

*   Public functions/classes: Full docstrings + Banners required.
*   Private functions (`_func`): Brief docstring required.

## 5. Inventory Maintenance

The `docs/architecture/mcp/mcp_operations_inventory.md` file tracks the implementation and testing status of every tool.

**When to Update:**
*   **Adding a Tool:** Add a new row with status `❌` (Untested) or `⚠️` (Partial).
*   **Deprecating a Tool:** Move to "Deprecated" section or mark as `[DEPRECATED]`.
*   **Verifying a Tool:** Update status to `✅` only after:
    1.  Unit tests pass.
    2.  Integration/Manual verification is complete.
    3.  Docstrings are compliant.

## 6. Versioning

*   Documentation updates should be committed alongside code changes.
*   Major architectural changes require updating `docs/architecture/mcp/architecture.md`.

## 7. Diagram Standards (ADR 085)

> **IMPORTANT**: Inline Mermaid blocks are **PROHIBITED** to prevent mnemonic bloat in learning snapshots.

### 7.1 The Canonical Pattern

All diagrams must follow this workflow:

1.  **Create `.mmd` file** in `docs/architecture_diagrams/{category}/`:
    ```
    docs/architecture_diagrams/
    ├── rag/           # RAG architecture
    ├── system/        # Infrastructure/fleet
    ├── transport/     # MCP transport
    └── workflows/     # Process/workflow
    ```

2.  **Add header metadata** to the `.mmd` file:
    ```text
    %% Name: My Diagram Title
    %% Description: What this diagram shows
    %% Location: docs/architecture_diagrams/category/my_diagram.mmd
    ```

3.  **Generate PNG** using the render script:
    ```bash
    python3 scripts/render_diagrams.py docs/architecture_diagrams/category/my_diagram.mmd
    ```

4.  **Reference in documents** with image AND source link:
    ```markdown
    ![Diagram Title](../../architecture_diagrams/workflows/protocol_128_learning_loop.png)
    
    *Source: [protocol_128_learning_loop.mmd](../../architecture_diagrams/workflows/protocol_128_learning_loop.mmd)*
    ```

### 7.2 Violation Detection

Use this command to find inline mermaid violations:
```bash
grep -rl '\`\`\`mermaid' . --include="*.md" | grep -v node_modules | grep -v .agent/learning/
```

**Reference**: [ADR 085](../../../ADRs/085_canonical_mermaid_diagram_management.md)
