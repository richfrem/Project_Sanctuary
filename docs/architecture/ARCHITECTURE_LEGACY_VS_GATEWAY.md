# MCP Server Architecture: Legacy vs Gateway Pattern

**Date:** 2024-12-24  
**Status:** Clarified  
**Reference:** ADR-066 v1.3, ADR-076

---

## Architecture Overview

Project Sanctuary uses a **dual-layer MCP architecture**:

1. **Gateway Clusters** (`mcp_servers/gateway/clusters/sanctuary_*`) - Containerized, SSE transport, connect to IBM ContextForge Gateway
2. **Operations Libraries** (`mcp_servers/*/operations.py`) - Business logic, shared by clusters
3. **Legacy Servers** (`mcp_servers/*/server.py`) - STDIO transport, for local/Claude Desktop use

---

## Current State Inventory

### Gateway Fleet (Containerized - SSE Transport)

| Cluster | Port | Tools | ADR-076 | Operations Library |
|---------|------|-------|---------|-------------------|
| sanctuary_utils | 8100 | 16 | ✅ | Various utility modules |
| sanctuary_filesystem | 8101 | 10 | ✅ | code/operations.py |
| sanctuary_network | 8102 | 2 | ✅ | Built-in |
| sanctuary_git | 8103 | 10 | ✅ | git/operations.py |
| sanctuary_cortex | 8104 | 13 | ✅ | rag_cortex/operations.py + forge_llm/operations.py |
| sanctuary_domain | 8105 | 35 | ✅ | chronicle, protocol, task, config, adr operations |

**Total Gateway Tools:** 86

### Legacy Servers (STDIO Transport - Claude Desktop)

| Server | Operations File | Gateway Equivalent | Recommended Action |
|--------|-----------------|-------------------|-------------------|
| adr | adr/operations.py | sanctuary_domain | Keep as ops library |
| agent_persona | agent_persona/operations.py | sanctuary_domain | Keep as ops library |
| chronicle | chronicle/operations.py | sanctuary_domain | Keep as ops library |
| code | code/operations.py | sanctuary_filesystem | Keep as ops library |
| config | config/operations.py | sanctuary_domain | Keep as ops library |
| council | council/operations.py | None (standalone) | Keep for CLI |
| forge_llm | forge_llm/operations.py | sanctuary_cortex | Keep as ops library |
| git | git/operations.py | sanctuary_git | Keep as ops library |
| orchestrator | orchestrator/ | None (meta-tool) | Keep for CLI |
| protocol | protocol/operations.py | sanctuary_domain | Keep as ops library |
| rag_cortex | rag_cortex/operations.py | sanctuary_cortex | Keep as ops library |
| task | task/operations.py | sanctuary_domain | Keep as ops library |
| workflow | workflow/ | None (workflow) | Keep for CLI |

---

## Architecture Pattern

```
mcp_servers/
├── gateway/
│   └── clusters/                   # MCP SERVER ENTRY POINTS
│       ├── sanctuary_cortex/
│       │   └── server.py          # Uses rag_cortex/operations.py
│       └── sanctuary_domain/
│           └── server.py          # Uses adr, chronicle, protocol, task, config/operations.py
├── rag_cortex/
│   ├── operations.py              # BUSINESS LOGIC (shared)
│   └── server.py                  # LEGACY (keep for STDIO/Claude Desktop)
└── lib/
    └── sse_adaptor.py             # SSEServer + @sse_tool decorator
```

---

## Recommendations

### DO NOT DELETE
- **operations.py files** - These are shared business logic used by Gateway clusters
- **Legacy server.py files** - These support STDIO transport for Claude Desktop integration

### DOCUMENTATION ONLY
- Legacy servers are **not deprecated**, they serve a different use case (local STDIO)
- Gateway clusters are the **production deployment** for SSE transport
- Both share the same operations libraries

### Future Consideration
- If Claude Desktop adds SSE support, legacy servers could be deprecated
- Until then, maintain dual-transport architecture per ADR-066 v1.3

---

*This document clarifies the architecture rather than deprecating legacy servers.*
