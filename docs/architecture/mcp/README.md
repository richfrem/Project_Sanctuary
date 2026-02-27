# Agent Plugin Integration Architecture Documentation

This folder contains the architectural documentation for Project Sanctuary's Agent Plugin Integration (Agent Plugin Integration) ecosystem.

---

## Quick Navigation

| Document | Description |
|----------|-------------|
| [[gateway_architecture|gateway_architecture.md]] | **Gateway Fleet** - 8 containers, 86 tools, dual-transport |
| [[mcp_ecosystem_architecture_v3|mcp_ecosystem_architecture_v3.md]] | **Complete Agent Plugin Integration Ecosystem** - All 15 domains (ADR 092) |
| [[legacy_mcp_architecture|legacy_mcp_architecture.md]] | **Legacy STDIO MCPs** - Original standalone servers |

---

## Architecture Overview

Project Sanctuary supports **two deployment modes**:

### 1. Gateway Mode (Recommended)
- Single `sanctuary_gateway` entry in Agent Plugin Integration client config
- Routes to 6 containerized clusters via IBM ContextForge
- **86 tools** across 8 containers
- See: [[gateway_architecture|gateway_architecture.md]]

### 2. Legacy Mode
- 12 separate Agent Plugin Integration servers in `.venv`
- Direct STDIO connections per server
- See: [[legacy_mcp_architecture|legacy_mcp_architecture.md]]

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
    └── class/                         # Agent Plugin Integration class diagrams (11 files)
```

---

## Key Decisions (ADRs)

| ADR | Title | Summary |
|-----|-------|---------|
| [[060_gateway_integration_patterns|060]] | Hybrid Fleet | 6 clusters + 6 guardrails |
| [[066_standardize_on_fastmcp_for_all_mcp_server_implementations|066]] | Dual-Transport | FastMCP STDIO + SSEServer for Gateway |
| [[076_sse_tool_metadata_decorator_pattern|076]] | @sse_tool Pattern | Decorator-based tool registration |

---

## Related Documentation

- [[README|Gateway Operations]] - Verification matrix & operations inventory
- [[README|Main README]] - Project overview with architecture diagrams
- [[PODMAN_OPERATIONS_GUIDE|Podman Guide]] - Fleet deployment instructions
