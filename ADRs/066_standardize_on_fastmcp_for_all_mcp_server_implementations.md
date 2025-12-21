# Standardize on FastMCP for All MCP Server Implementations

**Status:** proposed
**Date:** 2025-12-20
**Author:** Antigravity + User


---

## Context

During the implementation of ADR-065 (Unified Fleet CLI), we discovered that MCP servers using the custom `sse_adaptor.py` SSE implementation were failing tool discovery, while servers using the official FastMCP library worked correctly.

**Root Cause Analysis:**
- `sanctuary_domain` (FastMCP): 36 tools discovered ✅
- `sanctuary_cortex` (SSEServer): 11 tools discovered ⚠️
- `sanctuary_filesystem` (SSEServer): 3 tools discovered ⚠️
- `sanctuary_utils` (SSEServer): 0 tools discovered ❌
- `sanctuary_network` (SSEServer): 0 tools discovered ❌
- `sanctuary_git` (SSEServer): 0 tools discovered ❌

The IBM ContextForge Gateway expects strict MCP SSE protocol compliance:
1. SSE endpoint at `/sse` must emit proper JSON-RPC events
2. Messages endpoint at `/messages` must return 202 Accepted immediately
3. Response must be pushed through the active SSE stream with matching JSON-RPC ID
4. Event types must follow MCP specification

Our custom `sse_adaptor.py` implementation has inconsistent behavior that causes tool discovery failures for some servers.

## Decision

**All Sanctuary MCP servers MUST use FastMCP as the standard implementation library.**

1. **Mandatory Migration**: All existing servers using `sse_adaptor.py` must be migrated to FastMCP.

2. **New Server Template**: All new MCP servers must be scaffolded using FastMCP.

3. **Deprecate SSEServer**: The `mcp_servers/lib/sse_adaptor.py` will be deprecated and eventually removed.

4. **FastMCP Pattern**:
```python
from fastmcp import FastMCP

mcp = FastMCP("server_name")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description."""
    return result
```

5. **Migration Priority**:
   - Phase 1: sanctuary_git, sanctuary_utils, sanctuary_network (0 tools - broken)
   - Phase 2: sanctuary_cortex, sanctuary_filesystem (partially working)
   - Phase 3: Remove sse_adaptor.py entirely

## Consequences

**Positive:**
- Guaranteed MCP protocol compliance with IBM ContextForge Gateway
- Automatic tool schema generation via Python type hints
- Reduced maintenance burden (community-maintained library)
- Consistent developer experience across all servers
- Better error handling and debugging

**Negative:**
- Migration effort required for 5 servers
- FastMCP dependency must be added to all Dockerfiles
- Some custom capabilities in sse_adaptor.py may need alternative solutions

**Risks:**
- FastMCP version updates may introduce breaking changes
- Dependency on external library vs. internal control
