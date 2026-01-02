# Model Context Protocol (MCP) Documentation

**The Nervous System of Project Sanctuary**

## Overview

The Model Context Protocol (MCP) is the architectural backbone of Project Sanctuary, enabling a modular, "Nervous System" design. Instead of a monolithic application, the Sanctuary operates as a constellation of specialized servers that provide tools, resources, and intelligence to the central orchestrator and AI agents.

This architecture allows for:
- **Separation of Concerns:** Each server handles one domain (e.g., Git, Chronicle, RAG).
- **Scalability:** New capabilities can be added as new servers without modifying the core.
- **Interoperability:** Standardized protocol for tools and resources.
- **Security:** Granular control over what each agent can access.

## MCP Server Index

| Server | Domain | Documentation | Status |
|--------|--------|---------------|--------|
| **Cortex** | RAG, Memory, Semantic Search | [README](../../mcp_servers/cognitive/cortex/README.md) | ✅ Active |
| **Chronicle** | Historical Records, Truth | [README](../../mcp_servers/chronicle/README.md) | ✅ Active |
| **Protocol** | Doctrines, Laws | [README](../../mcp_servers/protocol/README.md) | ✅ Active |
| **Council** | Multi-Agent Orchestration | [README](../../mcp_servers/council/README.md) | ✅ Active |
| **Agent Persona** | Agent Roles & Dispatch | [README](../../mcp_servers/agent_persona/README.md) | ✅ Active |
| **Forge** | Fine-Tuning, Model Queries | [README](../../mcp_servers/system/forge/README.md) | ✅ Active |
| **Git Workflow** | Version Control, P101 v3.0 | [README](../../mcp_servers/system/git_workflow/README.md) | ✅ Active |
| **Task** | Task Management | [README](../../mcp_servers/task/README.md) | ✅ Active |
| **Code** | File I/O, Analysis | [README](../../mcp_servers/code/README.md) | ✅ Active |
| **Config** | System Configuration | [README](../../mcp_servers/config/README.md) | ✅ Active |
| **ADR** | Architecture Decisions | [README](../../mcp_servers/document/adr/README.md) | ✅ Active |

## Fleet Deployment & Management (ADR 065)

Project Sanctuary utilizes a unified **"Fleet of 8"** architecture, where core MCP servers (Cortex, Chronicle, Gateway, etc.) are deployed as a cohesive unit using container orchestration.

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
> 2. **SSE**: `SSEServer` or `MCP SDK` (Gateway).

See [ADR 066](../../ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md) for the mandatory transport selector pattern.
- **`make down`**: Safely stops the fleet.
- **`make restart`**: Restarts the fleet (or specific targets) and re-orchestrates.

> **Note:** Direct individual execution (e.g., `python3 -m ...`) is reserved for local development/debugging effectively "outside" the fleet mesh.

### The 3-Layer Fleet Pattern
1.  **Layer 1 (Physical):** Containers defined in `docker-compose.yml`. Managed by `podman`.
2.  **Layer 2 (Network):** Service discovery via `mcp_servers/gateway/fleet_resolver.py`.
3.  **Layer 3 (Logical):** Tool registration via `mcp_servers/gateway/fleet_orchestrator.py` -> `fleet_registry.json`.

For detailed architecture, see the [Gateway Architecture](../architecture/gateway/architecture/ARCHITECTURE.md).

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

- **Testing:** All MCP servers must follow the [Testing Standards](../standards/mcp/TESTING_STANDARDS.md).
- **Documentation:** Each server must have a README following the standard template.
- **Architecture:** See [MCP Ecosystem Overview](../architecture/mcp/diagrams/mcp_ecosystem_class.mmd).

## Related Resources

- [MCP Operations Inventory](../operations/mcp/mcp_operations_inventory.md) - Detailed list of all tools
- [RAG Strategies](../architecture/cortex/README.md) - Deep dive into Cortex architecture
- [Setup Guide](../operations/mcp/setup_guide.md) - Environment setup
