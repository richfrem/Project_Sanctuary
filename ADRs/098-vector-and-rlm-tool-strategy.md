# ADR-0097: Vector and RLM Tool Strategy

## Status
Accepted

## Date
2026-01-31

## Context
Project Sanctuary had accumulated two competing sets of tools for Vector Search and RLM (Recursive Language Model) operations:
1.  **Legacy (Integrated)**: `mcp_servers/rag_cortex` and `mcp_servers/learning`. Used by Protocol 128 (Learning Loop).
2.  **New (Standalone)**: `plugins/vector-db/` and `plugins/rlm-factory/`. Used by the new `generate_kit` workflow.

This duplication caused confusion about which tool to use for "Project Querying" vs "Tool Discovery". The standalone vector tools lacked critical features (like Parent-Child chunk hydration) present in the integrated Cortex RAG.

## Decision
We have decided to **Separated Concerns** between the "Cognitive Brain" and the "Tool Registry".

### 1. Vector Strategy (Reject Standalone)
*   **Decision**: We reject the standalone `plugins/vector-db/` implementation.
*   **Action**: The `plugins/vector-db/` directory has been deleted.
*   **Replacement**: All semantic search and RAG operations must use `mcp_servers/rag_cortex` (via `scripts/cortex_cli.py`). This ensures access to the full "Super-RAG" capabilities (Parent-Child, Time-Weighting).

### 2. RLM Strategy (Split Concerns)
We will maintain TWO distinct implementations of RLM for different domains:

*   **Cognitive RLM (Project Memory)**:
    *   **Manager**: `mcp_servers/learning/operations.py` (via `scripts/cortex_cli.py`).
    *   **Cache**: `.agent/learning/rlm_summary_cache.json`.
    *   **Purpose**: Manages the "Soul" of the project, Learning Loops, Debriefs, and Documentation Distillation.
    *   **Manifest**: `learning_manifest.json`.

*   **Tooling RLM (Tool Manuals)**:
    *   **Manager**: `plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py` (via `manage_tool_inventory.py`).
    *   **Cache**: `.agent/learning/rlm_tool_cache.json`.
    *   **Purpose**: Manages detailed metadata (headers, args, usage) for the `tool_inventory.json`.
    *   **Manifest**: `distiller_manifest.json` (Managed by Inventory).

### 3. Inventory Management
*   The `tool_inventory.json` descriptions are now Single Sourced from the `rlm_tool_cache.json`.
*   We use `distiller.py` (Tool Mode) to extract Extended Headers from python scripts and store them in the Tool Cache.
*   The Inventory then syncs with this Cache to ensure descriptions are accurate and high-fidelity.

## Consequences
*   **Pros**:
    *   Eliminates code duplication for Vector logic.
    *   Ensures consistent "Super-RAG" behavior for project queries.
    *   Provides high-quality, self-contained "Manual Pages" for tools via the Tool Cache.
*   **Cons**:
    *   Agents must know which RLM to query (Brain vs Tool). This is abstracted via the Agent's Tool definitions (`cortex_query` vs `fetch_tool_context`).

## Compliance
*   `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py` and `fetch_tool_context.py` default to "Tool Mode" for easy tool discovery.
*   `scripts/cortex_cli.py` defaults to "Legacy Mode" for project memory.
