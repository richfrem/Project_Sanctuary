# Agent Plugin Integration (Agent Plugin Integration) Documentation

**The Nervous System of Project Sanctuary**

## Overview

The Agent Plugin Integration (Agent Plugin Integration) is the architectural backbone of Project Sanctuary, enabling a modular, "Nervous System" design. Instead of a monolithic application, the Sanctuary operates as a constellation of specialized servers that provide tools, resources, and intelligence to the central orchestrator and AI agents.

This architecture allows for:
- **Separation of Concerns:** Each server handles one domain (e.g., Git, Chronicle, RAG).
- **Scalability:** New capabilities can be added as new servers without modifying the core.
- **Interoperability:** Standardized protocol for tools and resources.
- **Security:** Granular control over what each agent can access.

## Agent Plugin Integration Server Index

| Server | Domain | Documentation | Status |
|--------|--------|---------------|--------|
| **Cortex** | RAG, Memory, Semantic Search | [[README|README]] | ✅ Active |
| **Chronicle** | Historical Records, Truth | [[README|README]] | ✅ Active |
| **Protocol** | Doctrines, Laws | [[README|README]] | ✅ Active |
| **Council** | Multi-Agent Orchestration | [[README|README]] | ✅ Active |
| **Agent Persona** | Agent Roles & Dispatch | [[README|README]] | ✅ Active |
| **Forge** | Fine-Tuning, Model Queries | [[README|README]] | ✅ Active |
| **Git Workflow** | Version Control, P101 v3.0 | [[README|README]] | ✅ Active |
| **Task** | Task Management | [[README|README]] | ✅ Active |
| **Code** | File I/O, Analysis | [[README|README]] | ✅ Active |
| **Config** | System Configuration | [[README|README]] | ✅ Active |
| **ADR** | Architecture Decisions | [[README|README]] | ✅ Active |

## Fleet Deployment & Management (ADR 065)

Project Sanctuary utilizes a unified **"Fleet of 8"** architecture, where core Agent Plugin Integration servers (Cortex, Chronicle, Gateway, etc.) are deployed as a cohesive unit using container orchestration.

### Architecture Transition (Hybrid -> Unified)
Previously, the system operated in a **Hybrid Model (ADR 060)**:
*   **Containers (2)**: Only Heavy Infrastructure (`vector_db` via Chroma Image, `forge_llm` via Ollama Image) ran in Podman.
*   **Stdio (10)**: All other logic (Git, Chronicle, Task, etc.) ran as local Python processes via `start_mcp_servers.py`.

We have now moved to the **Unified Fleet Model (ADR 065)**:
*   **Containers (8)**: All core logic is containerized locally to ensure isolation and consistent networking.
*   **Stdio**: Reserved only for ad-hoc local debugging.

### Deployment Map (Dockerfiles)

| Logic Container | Dockerfile Source (Active) | Maps to Logical Server(s) | Legacy Source (Status) |
| :--- | :--- | :--- | :--- |
| **sanctuary_utils** | `gateway/clusters/sanctuary_utils` | `utils` | `mcp_servers/utils` (Deleted) |
| **sanctuary_filesystem** | `gateway/clusters/sanctuary_filesystem` | `code`, `filesystem` | `mcp_servers/code/Dockerfile` (Deleted) |
| **sanctuary_network** | `gateway/clusters/sanctuary_network` | `network` | `mcp_servers/network` (Deleted) |
| **sanctuary_git** | `gateway/clusters/sanctuary_git` | `git` | `mcp_servers/git/Dockerfile` (Deleted) |
| **sanctuary_cortex** | `gateway/clusters/sanctuary_cortex` | `rag_cortex` | `mcp_servers/rag_cortex/Dockerfile` (Deleted) |
| **sanctuary_domain** | `gateway/clusters/sanctuary_domain` | `chronicle`, `task`, `protocol`, `adr`, `config` | `mcp_servers/chronicle/Dockerfile` (Deleted) |
| **sanctuary-ollama** | *Image: `ollama/ollama:latest`* | `forge_llm` | N/A |
| **sanctuary_vector_db** | *Image: `chromadb/chroma:latest`* | `rag_cortex (db)` | N/A |

### The "Iron Root" Workflow
All fleet operations are centralized in the **Project Root Makefile**. This ensures idempotent, consistent deployment across environments.

**Key Commands:**
- **`make up`**: Deploys the full fleet (Physical) and registers tools (Logical).
- **`make status`**: Displays physical container health and logical registry state.

### Transport Standards (ADR 066)

> [!CAUTION]
> **FastMCP SSE is PROHIBITED** for Gateway connections. Fleet containers use a **Dual-Transport** architecture:
> 1. **STDIO**: FastMCP (Local/Claude Desktop).
> 2. **SSE**: `SSEServer` or `Agent Plugin Integration SDK` (Gateway).

See [[066_standardize_on_fastmcp_for_all_mcp_server_implementations|ADR 066]] for the mandatory transport selector pattern.
- **`make down`**: Safely stops the fleet.
- **`make restart`**: Restarts the fleet (or specific targets) and re-orchestrates.

> **Note:** Direct individual execution (e.g., `python3 -m ...`) is reserved for local development/debugging effectively "outside" the fleet mesh.

### The 3-Layer Fleet Pattern
1.  **Layer 1 (Physical):** Containers defined in `docker-compose.yml`. Managed by `podman`.
2.  **Layer 2 (Network):** Service discovery via `mcp_servers/gateway/fleet_resolver.py`.
3.  **Layer 3 (Logical):** Tool registration via `mcp_servers/gateway/fleet_orchestrator.py` -> `fleet_registry.json`.

For detailed architecture, see the [[ARCHITECTURE|Gateway Architecture]].

### Tool Federation Status (Updated 2025-12-20)

All 6 logic containers successfully federate **84 tools** to the IBM ContextForge Gateway:

| Container | Tool Count | Status |
|-----------|------------|--------|
| sanctuary_cortex | 11 | ✅ |
| sanctuary_domain | 36 | ✅ |
| sanctuary_filesystem | 10 | ✅ |
| sanctuary_git | 9 | ✅ |
| sanctuary_network | 2 | ✅ |
| sanctuary_utils | 16 | ✅ |
| **TOTAL** | **84** | ✅ |

**Verification Command:**
```bash
python3 -m mcp_servers.gateway.gateway_client tools -v
```

## Development Standards

- **Testing:** All Agent Plugin Integration servers must follow the [[TESTING_STANDARDS|Testing Standards]].
- **Documentation:** Each server must have a README following the standard template.
- **Architecture:** See [[mcp_ecosystem_architecture_v3.mmd|Agent Plugin Integration Ecosystem Overview]].

## Related Resources

- [[mcp_operations_inventory|Agent Plugin Integration Operations Inventory]] - Detailed list of all tools
- [[README|RAG Strategies]] - Deep dive into Cortex architecture
- [[setup_guide|Setup Guide]] - Environment setup
