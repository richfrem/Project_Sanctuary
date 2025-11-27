# MCP Port Registry

**Version:** 1.0  
**Status:** Active  
**Purpose:** Centralized registry of port assignments for Project Sanctuary MCP servers to prevent conflicts.

---

## Port Allocation Strategy

- **Range:** 3000-3099
- **Protocol:** HTTP (SSE) / Stdio (No port needed)
- **Container Mapping:** Host Port -> Container Port (8080)

## Assigned Ports

| Port | Server Name | Domain | Status |
|------|-------------|--------|--------|
| **3001** | Chronicle MCP | `project_sanctuary.document.chronicle` | Planned |
| **3002** | Protocol MCP | `project_sanctuary.document.protocol` | Planned |
| **3003** | ADR MCP | `project_sanctuary.document.adr` | Planned |
| **3004** | **Task MCP** | `project_sanctuary.document.task` | **Active** |
| **3005** | RAG MCP (Cortex) | `project_sanctuary.cognitive.cortex` | Planned |
| **3006** | Council MCP | `project_sanctuary.cognitive.council` | Planned |
| **3007** | Config MCP | `project_sanctuary.system.config` | Planned |
| **3008** | Code MCP | `project_sanctuary.system.code` | Planned |
| **3009** | Git Workflow MCP | `project_sanctuary.system.git_workflow` | Planned |
| **3010** | Forge MCP | `project_sanctuary.model.fine_tuning` | Planned |

## Usage

When running a container, map the assigned host port to the container's internal port (usually 8080).

**Example (Task MCP):**
```bash
podman run -p 3004:8080 ...
```
