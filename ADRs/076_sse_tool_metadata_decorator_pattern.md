# ADR 076: SSE Tool Metadata Decorator Pattern

**Status:** ✅ ACCEPTED
**Red Team Review:** Approved with hardening recommendations
**Date:** 2025-12-24
**Deciders:** Antigravity, User
**Technical Story:** Gateway tool descriptions missing - all 85 tools registered with "No description"

---

## Context

During Gateway fleet verification, we discovered that **all 85 federated tools** are registered without descriptions in the IBM ContextForge Gateway admin UI. This degrades LLM tool discovery and makes fleet management difficult.

### Root Cause

The SSEServer's `register_tool()` method extracts descriptions from `handler.__doc__`:

```python
# mcp_servers/lib/sse_adaptor.py (line 88-93)
def register_tool(self, name: str, handler: Callable[..., Awaitable[Any]], schema: Optional[Dict] = None):
    self.tools[name] = {
        "handler": handler,
        "schema": schema,
        "description": handler.__doc__.strip() if handler.__doc__ else "No description"
    }
```

However, the SSE wrapper functions in fleet servers lack docstrings:

```python
# PROBLEM: No docstring on SSE wrapper
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)  # ← Missing docstring!
    return json.dumps({"status": "success", "debrief": response}, indent=2)
```

Meanwhile, FastMCP (STDIO) versions have docstrings via the `@mcp.tool()` decorator pattern:

```python
# WORKING: FastMCP has docstrings
@mcp.tool()
async def cortex_learning_debrief(request: CortexLearningDebriefRequest) -> str:
    """Scans repository for technical state changes (Protocol 128)."""  # ← Has docstring
    ...
```

**Result:** STDIO mode gets descriptions; SSE mode gets "No description".

### Scope

- **Affected:** 6 fleet containers (sanctuary_utils, filesystem, network, git, cortex, domain)
- **Tools affected:** 85 total
- **Impact:** LLM tool discovery degraded, admin UI shows "No description" for all tools

### Alignment with ADR 066 (Dual-Transport Architecture)

Per [ADR 066](./066_standardize_on_fastmcp_for_all_mcp_server_implementations.md), Project Sanctuary uses a **dual-transport standard**:

| Transport | Implementation | Decorator | Use Case |
|-----------|---------------|-----------|----------|
| **STDIO** | FastMCP | `@mcp.tool()` | Claude Desktop, IDE direct |
| **SSE** | SSEServer | `@sse_tool()` *(proposed)* | Gateway Fleet containers |

This ADR proposes `@sse_tool()` as the **SSE-transport counterpart** to FastMCP's `@mcp.tool()`:

![MCP Tool Decorator Pattern](../../docs/architecture_diagrams/system/mcp_tool_decorator_pattern.png)

*[Source: mcp_tool_decorator_pattern.mmd](../../docs/architecture_diagrams/system/mcp_tool_decorator_pattern.mmd)*

**Key Alignment Points:**
- Both decorators attach metadata at function definition site
- Both support explicit `name`, `description`, and `schema`
- Both delegate to shared `operations.py` (3-Layer Architecture per ADR 066)
- `@sse_tool()` is for SSE mode **only** — FastMCP's `@mcp.tool()` remains unchanged for STDIO

---

## Alternatives Considered

### Option 1: Add Docstrings to SSE Wrapper Functions (Quick Fix)

**Description:** Simply add docstrings to each SSE wrapper function.

```python
def cortex_learning_debrief(hours: int = 24):
    """Scans repository for technical state changes (Protocol 128)."""
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)
```

**Pros:**
- Minimal change
- Works immediately with existing SSEServer

**Cons:**
- Duplicates docstrings between STDIO and SSE implementations
- No explicit metadata - description hidden in docstring
- Easy to forget when adding new tools

**Why not chosen:** Maintainability concerns with duplication across ~85 tools.

---

### Option 2: Centralized Tool Registry File

**Description:** Create a JSON or Python dict with all tool metadata.

```python
# tool_registry.py
TOOL_METADATA = {
    "cortex_learning_debrief": {
        "description": "Scans repository for technical state changes (Protocol 128).",
        "schema": LEARNING_DEBRIEF_SCHEMA
    },
    ...
}
```

**Pros:**
- Single source of truth
- Easy to export/import
- Separates metadata from implementation

**Cons:**
- Another file to maintain
- Metadata disconnected from function definition
- Easy for registry and code to drift

**Why not chosen:** Increases maintenance burden and risk of drift.

---

### Option 3: Inherit from Operations Layer

**Description:** Pull docstrings from the operations layer methods.

```python
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)

# Inherit docstring from operations
cortex_learning_debrief.__doc__ = CortexOperations.learning_debrief.__doc__
```

**Pros:**
- Single source of truth in operations layer
- No duplication

**Cons:**
- Requires operations methods to have docstrings (not always the case)
- Awkward post-hoc assignment
- Not explicit at function definition site

**Why not chosen:** Requires refactoring operations layer first; awkward pattern.

---

### Option 4: Decorator Pattern with `@sse_tool` (RECOMMENDED) ⭐

**Description:** Create a decorator similar to FastMCP's `@mcp.tool()` that attaches metadata to functions.

```python
@sse_tool(
    name="cortex_learning_debrief",
    description="Scans repository for technical state changes (Protocol 128).",
    schema=LEARNING_DEBRIEF_SCHEMA
)
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)
```

**Pros:**
- Explicit metadata at function definition site
- Consistent with FastMCP's `@mcp.tool()` pattern
- Hard to forget - decorator is required for registration
- Enables auto-registration of decorated functions
- Single source of truth (decorator params)

**Cons:**
- Requires changes to SSEServer
- Slightly more boilerplate than Option 1

---

## Decision

Adopt **Option 4: Decorator Pattern with `@sse_tool`**.

### Implementation

#### 1. Add decorator to `sse_adaptor.py`

```python
# mcp_servers/lib/sse_adaptor.py

def sse_tool(
    name: str = None, 
    description: str = None, 
    schema: dict = None
):
    """
    Decorator to mark functions as SSE tools with explicit metadata.
    
    Usage:
        @sse_tool(
            name="cortex_query",
            description="Perform semantic search query.",
            schema=QUERY_SCHEMA
        )
        def cortex_query(query: str, max_results: int = 5):
            ...
    """
    def decorator(func):
        func._sse_tool = True
        func._sse_name = name or func.__name__
        func._sse_description = description or func.__doc__ or "No description"
        func._sse_schema = schema or {"type": "object", "properties": {}}
        return func
    return decorator
```

#### 2. Add auto-registration method to SSEServer

```python
class SSEServer:
    # ... existing code ...
    
    def register_decorated_tools(self, namespace: dict):
        """
        Auto-register all functions decorated with @sse_tool.
        
        Usage:
            server.register_decorated_tools(locals())
        """
        for name, obj in namespace.items():
            if callable(obj) and getattr(obj, '_sse_tool', False):
                self.register_tool(
                    name=obj._sse_name,
                    handler=obj,
                    schema=obj._sse_schema,
                    description=obj._sse_description
                )
```

#### 3. Update `register_tool` to accept explicit description

```python
def register_tool(
    self, 
    name: str, 
    handler: Callable[..., Awaitable[Any]], 
    schema: Optional[Dict] = None,
    description: str = None  # NEW: explicit parameter
):
    self.tools[name] = {
        "handler": handler,
        "schema": schema,
        "description": description or handler.__doc__.strip() if handler.__doc__ else "No description"
    }
    self.logger.info(f"Registered tool: {name}")
```

#### 4. Update fleet servers to use decorator

**Before (sanctuary_cortex/server.py):**
```python
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)

# Manual registration
server.register_tool("cortex_learning_debrief", cortex_learning_debrief, LEARNING_DEBRIEF_SCHEMA)
```

**After:**
```python
@sse_tool(
    name="cortex_learning_debrief",
    description="Scans repository for technical state changes (Protocol 128).",
    schema=LEARNING_DEBRIEF_SCHEMA
)
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)

# Auto-registration
server.register_decorated_tools(locals())
```

---

## Consequences

### Positive
- **Explicit metadata:** Name, description, and schema defined at function site
- **Pattern parity:** Consistent with FastMCP's `@mcp.tool()` decorator
- **Auto-registration:** Reduces boilerplate and prevents forgotten registrations
- **Single source of truth:** Metadata lives with the function definition
- **Backward compatible:** Existing `register_tool()` still works

### Negative
- **SSEServer changes:** Requires updates to `sse_adaptor.py`
- **Server updates:** All 6 fleet servers need decorator migration
- **Container rebuild:** All fleet containers must be rebuilt

### Risks
- **Migration errors:** Could break existing tools if refactoring is incomplete
- **Mitigation:** Test each server after migration before Gateway re-registration

### Dependencies
- ADR 066 (Dual-Transport Standards) - Complements this decision
- All 6 fleet server modules

---

## Implementation Notes

### Migration Checklist

1. [ ] Update `mcp_servers/lib/sse_adaptor.py` with decorator and auto-registration
2. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_cortex/server.py`
3. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_domain/server.py`
4. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_utils/server.py`
5. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_git/server.py`
6. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_filesystem/server.py`
7. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_network/server.py`
8. [ ] Rebuild all fleet containers
9. [ ] Re-run fleet setup: `python -m mcp_servers.gateway.fleet_setup`
10. [ ] Verify descriptions in Gateway admin UI

### Verification

```bash
# After implementation, verify descriptions appear:
curl -s https://localhost:4444/api/tools -H "Authorization: Bearer $TOKEN" | jq '.[].description'

# Verify SSE handshake still works (must return 'endpoint' event immediately):
curl -N http://localhost:8100/sse
```

---

## Red Team Hardening Recommendations

> [!IMPORTANT]
> **The following hardening measures are MANDATORY per Red Team review.**

### 1. Namespace Safety

`register_decorated_tools()` must ignore private functions (starting with `_`) to prevent accidental registration:

```python
def register_decorated_tools(self, namespace: dict):
    for name, obj in namespace.items():
        if name.startswith('_'):  # Skip private functions
            continue
        if callable(obj) and getattr(obj, '_sse_tool', False):
            self.register_tool(...)
```

### 2. Strict Schema Linking

The `schema` parameter should reference Pydantic-generated schemas from `models.py` to ensure SSE and FastMCP wrappers validate against identical constraints.

### 3. Handshake Verification

After migration, verify that `register_decorated_tools()` does not interfere with the immediate `endpoint` event response required by the Gateway (curl check above).

---

## Red Team Sign-Off

| Reviewer | Verdict | Date |
|----------|---------|------|
| User | ✅ APPROVED | 2025-12-24 |
| Red Team Analysis | ✅ APPROVED with hardening | 2025-12-24 |

---

## Future Considerations

- **Schema validation:** Decorator could validate schemas at decoration time
- **Type inference:** Could auto-generate schemas from type hints
- **Documentation generation:** Decorated metadata could feed API docs

---

## References

- [ADR 066: MCP Server Transport Standards](./066_standardize_on_fastmcp_for_all_mcp_server_implementations.md)
- [sse_adaptor.py](../mcp_servers/lib/sse_adaptor.py)
- [FastMCP Decorator Pattern](https://github.com/jlowin/fastmcp)
