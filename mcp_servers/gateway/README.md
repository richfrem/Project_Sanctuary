# Sanctuary Gateway Module
Location: `mcp_servers/gateway/`

## Overview
This module contains the primary client and management interfaces for the Sanctuary Gateway. It is responsible for server registration, tool discovery, and high-level fleet orchestration using the **3-Layer Declarative Fleet Pattern**.

## Prerequisites
Before interacting with the Gateway, ensure the following infrastructure is operational:
- **Podman**: The "Fleet of 8" runs in isolated containers via Podman.
- **Docker Compose**: Managed via [docker-compose.yml](../../docker-compose.yml).
- **Network**: The external Gateway service (`mcp_gateway`) MUST be connected to the fleet network (`mcp_network`):
  ```bash
  podman network connect mcp_network mcp_gateway
  ```
- **Environment**: Root `.env` must define `MCP_GATEWAY_URL` and `MCPGATEWAY_BEARER_TOKEN`.

## Primary Components
The fleet management is organized into three distinct layers to separate intent from execution:

1. **[fleet_spec.py](fleet_spec.py)**: **Layer 1 (Intent)** - Canonical identities and default endpoints for the fleet.
2. **[fleet_resolver.py](fleet_resolver.py)**: **Layer 2 (Policy)** - Reconciles spec with runtime environment variables.
3. **[fleet_orchestrator.py](fleet_orchestrator.py)**: **Layer 3 (Persistence)** - Drives discovery handshakes and persists state to `fleet_registry.json`.
4. **[gateway_client.py](gateway_client.py)**: **Pure Transport** - Fleet-aware library for registrations and JSON-RPC execution.

## Operational Workflow (ADR 065)

**This module is now orchestrated by the Project Root Makefile.**
Manual execution of `podman compose` or `fleet_orchestrator.py` is generally not needed for day-to-day operations.

### Unified Command
To build, deploy, and register this Gateway module alongside the entire fleet, simply run from the project root:

```bash
make up
```

This command automatically:
1. Starts the infrastructure (`podman compose`).
2. Waits for health checks (`wait_for_pulse.sh`).
3. Runs the discovery logic (`fleet_orchestrator.py`).
4. Updates the registry (`fleet_registry.json`).

For status monitoring:
```bash
make status
```

*See [Project Root Makefile](../../Makefile) for implementation details.*

## Adding a New Operation to the Fleet

This guide documents the steps to add a new MCP tool to an existing fleet server (e.g., `sanctuary_cortex`).

### Step 1: Define Models (`models.py`)

Add request/response dataclasses and Pydantic models:

```python
# In mcp_servers/rag_cortex/models.py

@dataclass
class MyNewOperationRequest:
    param1: str
    param2: int = 0

@dataclass  
class MyNewOperationResponse:
    status: str
    result: str
    error: Optional[str] = None

# FastMCP Pydantic model (for STDIO transport)
class CortexMyNewOperationRequest(BaseModel):
    param1: str = Field(..., description="First parameter")
    param2: int = Field(0, description="Second parameter")
```

### Step 2: Implement Operation (`operations.py`)

Add the method to the operations class:

```python
# In mcp_servers/rag_cortex/operations.py

def my_new_operation(self, request: MyNewOperationRequest) -> MyNewOperationResponse:
    try:
        # Your logic here
        return MyNewOperationResponse(status="success", result="done")
    except Exception as e:
        return MyNewOperationResponse(status="error", result="", error=str(e))
```

### Step 3: Expose Tool (`server.py`)

Add both SSE and STDIO tool definitions:

```python
# In mcp_servers/gateway/clusters/sanctuary_cortex/server.py

# 1. Add schema (for SSE transport)
MY_NEW_OPERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "param1": {"type": "string", "description": "First parameter"},
        "param2": {"type": "integer", "description": "Second parameter"}
    },
    "required": ["param1"]
}

# 2. Add SSE tool (inside run_sse_server)
@sse_tool(
    name="cortex_my_new_operation",
    description="Description of what this tool does.",
    schema=MY_NEW_OPERATION_SCHEMA
)
def cortex_my_new_operation(param1: str, param2: int = 0):
    from mcp_servers.rag_cortex.models import MyNewOperationRequest
    request = MyNewOperationRequest(param1=param1, param2=param2)
    response = get_ops().my_new_operation(request)
    return json.dumps(to_dict(response), indent=2)

# 3. Add STDIO tool (inside run_stdio_server)
@mcp.tool()
async def cortex_my_new_operation(request: CortexMyNewOperationRequest) -> str:
    from mcp_servers.rag_cortex.models import MyNewOperationRequest
    internal_req = MyNewOperationRequest(param1=request.param1, param2=request.param2)
    response = get_ops().my_new_operation(internal_req)
    return json.dumps(to_dict(response), indent=2)

# 4. Add import (STDIO section)
from mcp_servers.rag_cortex.models import CortexMyNewOperationRequest
```

> **Important**: SSE tools must use regular `def`, not `async def`.

### Step 4: Rebuild & Restart Container

```bash
# Rebuild the container with new code
podman compose -f docker-compose.yml build sanctuary_cortex

# Restart the service
podman compose restart sanctuary_cortex
```

### Step 5: Re-register with Gateway

```bash
# Full fleet re-registration (recommended)
python3 -m mcp_servers.gateway.fleet_setup

# Or single-server re-registration
python3 -m mcp_servers.gateway.fleet_setup --server sanctuary_cortex --no-clean
```

### Step 6: Verify Tool Registration

```bash
# Check tool count (should increment)
python3 -m mcp_servers.gateway.gateway_client tools --server sanctuary_cortex

# View registry
cat mcp_servers/gateway/fleet_registry.json | jq '.fleet_servers.cortex.tools | length'
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Tool count unchanged | Ensure SSE tool uses `def` not `async def` |
| Container won't start | Check `podman compose logs sanctuary_cortex` |
| Import errors | Verify model imports in both SSE and STDIO sections |
| Gateway shows old count | Run full `fleet_setup` without `--no-clean` |

## Documentation
- **[Gateway Architecture](../../docs/mcp/servers/gateway/architecture/ARCHITECTURE.md)**: Deep dive into the 3-layer pattern.
- **[ADR 064](../../ADRs/064_centralized_registry_for_fleet_of_8_mcp_servers.md)**: Design decision for centralized registry.
- **[Podman Operations Guide](../../docs/PODMAN_OPERATIONS_GUIDE.md)**: Detailed container management instructions.
