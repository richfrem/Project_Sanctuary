# MCP Architecture Documentation

This folder contains the architectural documentation for Project Sanctuary's MCP (Model Context Protocol) ecosystem.

---

## Quick Navigation

| Document | Description |
|----------|-------------|
| [gateway_architecture.md](./gateway_architecture.md) | **Gateway Fleet** - 8 containers, 86 tools, dual-transport |
| [mcp_ecosystem_architecture_v3.md](./mcp_ecosystem_architecture_v3.md) | **Complete MCP Ecosystem** - All 15 domains (ADR 092) |
| [legacy_mcp_architecture.md](./legacy_mcp_architecture.md) | **Legacy STDIO MCPs** - Original standalone servers |

---

## Architecture Overview

Project Sanctuary supports **two deployment modes**:

### 1. Gateway Mode (Recommended)
- Single `sanctuary_gateway` entry in MCP client config
- Routes to 6 containerized clusters via IBM ContextForge
- **86 tools** across 8 containers
- See: [gateway_architecture.md](./gateway_architecture.md)

### 2. Legacy Mode
- 12 separate MCP servers in `.venv`
- Direct STDIO connections per server
- See: [legacy_mcp_architecture.md](./legacy_mcp_architecture.md)

---

## Folder Structure

```
architecture/
├── README.md                          # This file
├── gateway_architecture.md            # Gateway Fleet docs
├── mcp_ecosystem_architecture_v3.md   # Full ecosystem spec
├── legacy_mcp_architecture.md         # Legacy STDIO servers
│
├── analysis/                          # Design analysis docs
│   └── ddd_analysis.md               # Domain-Driven Design
│
└── diagrams/                          # Mermaid diagrams
    ├── architecture/                  # System-level (7 files)
    ├── transport/                     # STDIO/SSE flows (3 files)
    ├── workflows/                     # Process diagrams (6 files)
    └── class/                         # MCP class diagrams (11 files)
```

---

## Key Decisions (ADRs)

| ADR | Title | Summary |
|-----|-------|---------|
| [060](../../../ADRs/060_gateway_integration_patterns.md) | Hybrid Fleet | 6 clusters + 6 guardrails |
| [066](../../../ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md) | Dual-Transport | FastMCP STDIO + SSEServer for Gateway |
| [076](../../../ADRs/076_sse_tool_metadata_decorator_pattern.md) | @sse_tool Pattern | Decorator-based tool registration |

---

## Related Documentation

- [Gateway Operations](servers/gateway/operations/README.md) - Verification matrix & operations inventory
- [Main README](../../../README.md) - Project overview with architecture diagrams
- [Podman Guide](../../../docs/operations/processes/PODMAN_OPERATIONS_GUIDE.md) - Fleet deployment instructions
