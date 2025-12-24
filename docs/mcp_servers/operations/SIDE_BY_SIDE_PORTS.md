# Side-by-Side Architecture: Port Management Strategy

## Objective
Support two distinct deployment modes without conflict:
1.  **Legacy/Direct Mode:** Standard MCP servers running locally (scripts).
2.  **Gateway Mode:** Containerized "Fleet of 8" routed via Sanctuary Gateway.

## Definitive Port Source of Truth (Canonical)

### 1. Unified Entry Point
| Service | Host Port | Protocol | Usage |
| :--- | :--- | :---: | :--- |
| **Sanctuary Gateway** | `4444` | HTTPS/SSE | Primary Agent Entry Point |

### 2. The Fleet of 8 (Containerized)
To prevent conflicts with legacy services (8000-8099), the Fleet is pinned to the **81xx range**.

| Service | Host Port | Internal Port | Cluster |
| :--- | :---: | :---: | :--- |
| **sanctuary_utils** | `8100` | `8000` | Utils |
| **sanctuary_filesystem** | `8101` | `8000` | SysAdmin |
| **sanctuary_network** | `8102` | `8000` | External |
| **sanctuary_git** | `8103` | `8000` | SysAdmin |
| **sanctuary_cortex** | `8104` | `8000` | Intelligence |
| **sanctuary_domain** | `8105` | `8105` | Business/Domain |
| **sanctuary_vector_db** | `8110` | `8000` | Knowledge Backend |
| **sanctuary_ollama_mcp**| `11434` | `11434` | Model Backend |

### 3. Legacy & External Services (Script-based)
| Service | Host Port | Range | Usage |
| :--- | :---: | :---: | :--- |
| **helloworld_mcp** | `8005` | Legacy | Isolated Test Server |
| **Legacy Cortex** | `8090` | Reserved | Non-containerized script |
| **Legacy Filesystem** | `8091` | Reserved | Non-containerized script |
| **Legacy Domain** | `8092` | Reserved | Non-containerized script |

## Conflict Resolution Rules (Hard Enforcement)
1.  **FLEET PROTECTED RANGE:** Host ports `8100-8110` are exclusively for Podman Fleet containers.
2.  **LEGACY RANGE:** Host ports `8090-8099` are for legacy scripts to avoid `8000` (common dev port).
3.  **HELLOWORLD ISOLATION:** Port `8005` is exclusively for the `helloworld_mcp` debug server.
4.  **NO DUAL MAPPING:** No container shall map multiple host ports (Fixed `sanctuary_vector_db` 8000/8005 error).
5.  **DOCKER-COMPOSE SUPREMACY:** `docker-compose.yml` hardcodes these ports to override any `.env` desync.

## Implementation Status
- [x] `SIDE_BY_SIDE_PORTS.md` (Source of Truth)
- [ ] `docker-compose.yml` (Hardcoded mappings)
- [ ] `README.md` (Architectural diagrams)
- [ ] `docs/Protocol_056_MCP_Architecture_Analysis.md` (Diagram update)
- [ ] `docs/PODMAN_STARTUP_GUIDE.md` (Quick reference update)
- [ ] `.env` and `.env.example` synchronization
