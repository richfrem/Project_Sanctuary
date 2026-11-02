# ADR 092: Transition to 15 Canonical MCP Servers

## Status
ACCEPTED

## Context
Project Sanctuary's architecture has evolved significantly since the initial "Fleet of 8" (ADR 063) and the "Canonical 12" (ADR 066). We have identified two major areas of architectural friction:

1.  **RAG Cortex Overloading**: The `mcp_servers/rag_cortex` server, ostensibly dedicated to Vector DB and Semantic Search, has become a "dumping ground" for System Lifecycle tools (`capture_snapshot`, `learning_debrief`, `guardian_wakeup`, `persist_soul`). This violates the Single Responsibility Principle.
2.  **Orphaned Evolutionary Logic**: New capabilities derived from Digital Red Queen (DRQ) research (Protocol 131), such as Map-Elites metrics (`metrics.py`), lack a permanent home, currently residing in the `LEARNING/` directory.
3.  **The "Hidden" Server**: `mcp_servers/workflow` exists and is functional but is not officially recognized in the canonical list or documentation.

To align with Protocol 128 (Recursive Learning Loop) and Protocol 131 (Evolutionary Self-Improvement), we need to elevate these concerns to first-class citizens in the architecture.

## Decision
We will transition from the "Canonical 12" to the **"Canonical 15" MCP Servers**, organizing them into clear logical domains.

### 1. Official Recognition of Workflow MCP
**`mcp_servers/workflow`** is formally recognized as the 13th Canonical Server.
*   **Role**: Standard Operating Procedures (SOPs) discovery and retrieval.

### 2. Creation of Learning MCP (Protocol 128)
We will create **`mcp_servers/learning`** (The 14th Server).
*   **Role**: Enforces Protocol 128 (Cognitive Continuity). Manages the Session Lifecycle.
*   **Migrated Operations** (from `rag_cortex`):
    *   `learning_debrief` (The Scout)
    *   `capture_snapshot` (The Seal)
    *   `persist_soul` (The Chronicle)
    *   `guardian_wakeup` (The Bootloader)

### 3. Creation of Evolution MCP (Protocol 131)
We will create **`mcp_servers/evolution`** (The 15th Server).
*   **Role**: Enforces Protocol 131 (Self-Improvement). Manages Mutation and Selection.
*   **New Operations**:
    *   `measure_complexity`: Computes Map-Elites axes (migrated from `metrics.py`).
    *   (Future) `validate_mutation`
    *   (Future) `register_edge_case`

### 4. Purification of RAG Cortex
**`mcp_servers/rag_cortex`** will return to its purely distinct role:
*   **Role**: High-performance Vector Database interaction, Semantic Search, and Memory Retrieval.

## Revised Architecture Map (The Canonical 15)

| ID | Server | Directory | Theme | Cluster |
|----|--------|-----------|-------|---------|
| 1 | Chronicle | `chronicle/` | Truth/History | `domain` |
| 2 | Protocol | `protocol/` | Governance | `domain` |
| 3 | ADR | `adr/` | Decision Records | `domain` |
| 4 | Task | `task/` | Work Tracking | `domain` |
| 5 | Git | `git/` | Repo State | `git` |
| 6 | RAG Cortex | `rag_cortex/` | Semantic Memory | `cortex` |
| 7 | Forge LLM | `forge_llm/` | Inference | `cortex` |
| 8 | Config | `config/` | Configuration | `utils` |
| 9 | Code | `code/` | Filesystem/Analysis | `filesystem` |
| 10 | Agent Persona | `agent_persona/` | Identity | `cortex` |
| 11 | Council | `council/` | Deliberation | `cortex` |
| 12 | Orchestrator | `orchestrator/` | Strategy/Planning | `cortex` |
| 13 | Workflow | `workflow/` | SOPs/Process | `domain` |
| 14 | **Learning** | `learning/` | Lifecycle (P128) | `cortex` |
| 15 | **Evolution** | `evolution/` | Improvement (P131) | `cortex` |

## Consequences
*   **Better Separation of Concerns**: Lifecycle implementation details no longer pollute the Vector DB logic.
*   **Clearer Evolution Path**: Protocol 131 has a dedicated "lab" (`evolution` server) to grow without risking system stability.
*   **Refactoring Required**: `rag_cortex/operations.py` must be split. `cortex_cli.py` must be updated to import from new locations.
*   **Fleet Updates**: `fleet_spec.py` and `fleet_registry.json` must serve these new tools, likely via the `sanctuary_cortex` cluster.

## Implementation Strategy
1.  **Scaffold**: Create new server directories using the standard FastMCP template.
2.  **Migrate**: Move code from `rag_cortex` and `metrics.py`.
3.  **Expose**: Register new tools in `sanctuary_cortex/server.py` (or a new cluster if deemed necessary, though `cortex` is the logical fit).
4.  **Document**: Update `mcp_operations_inventory.md`.
