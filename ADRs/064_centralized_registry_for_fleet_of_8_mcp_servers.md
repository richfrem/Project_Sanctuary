# Centralized Registry for Fleet of 8 MCP Servers

**Status:** approved
**Date:** 2025-12-20
**Author:** user and AI Assistant and redteam (GPT5.2)
**Version:** 1.5


---

## Context

The Sanctuary Gateway oversees a specialized set of MCP servers known as the 'Fleet of 8'. These servers require systematic registration, initialization, and tool discovery. 

### The Shift: "Input" vs. "Output"

| Feature | Legacy Approach (Input JSON) | 3-Layer Pattern (Output JSON) |
| :--- | :--- | :--- |
| **Source of Truth** | Static strings in JSON file | Python `FLEET_SPEC` (Intent) |
| **URL Management** | Hardcoded in JSON | `Resolver` (reconciles Spec + Env) |
| **Tool Inventory** | Manually maintained | **Auto-populated** via Handshake |
| **JSON Purpose** | Direct configuration | **Discovery Manifest** (Documentation) |

A previous attempt to manage these definitions via a standalone JSON configuration file was deemed incorrect due to:
1.  **Inflexible**: Lack of logic handling (e.g., conditional initialization).
2.  **Import Path Fragility**: Difficulty in sharing JSON paths across local execution and container environments.
3.  **Synchronization Latency**: Static files quickly falling out of sync with code-driven bridge logic.

## Decision

We will adopt a **3-Layer Declarative Fleet Pattern**: "Code-Defined Intent, Runtime-Resolved Reality." This decouples topology from transport logic.

### 1. The Spec Layer (Intent)
A pure Python data model defining the cluster identities (Slugs, SSE mappings).
- Resides in: `fleet_spec.py`.
- Purpose: Authoritative Design Intent.

### 2. The Resolver Layer (Policy)
Logic that reconciles the **Spec** with the **Runtime Context** (Environment Variables, Docker settings).
- Resides in: `fleet_resolver.py`.
- Purpose: Determining the final "Ready-to-Connect" endpoint.

### 3. The Observation Layer (Runtime State)
What the Gateway discovers during handshakes (Tools, Schemas).
- Resides in: `fleet_registry.json`.
- Purpose: UI/AI Discovery Manifest. **Crucially, core logic never reads this file for logic.**

## Architectural Flow

![mcp_fleet_resolution_flow](docs/architecture_diagrams/system/mcp_fleet_resolution_flow.png)

*[Source: mcp_fleet_resolution_flow.mmd](docs/architecture_diagrams/system/mcp_fleet_resolution_flow.mmd)*

## Consequences

- **Separation of Concerns**: `gateway_client.py` becomes a pure transport library.
- **Explainability**: Clear hierarchy of "Why is the system using this URL?" (Spec < DB < Env).
- **Testability**: Tests can inject mock resolvers without spawning real containers.
- **Asynchronous Resilience**: Handshakes are scheduled observations, not blocking boot-time requirements.
- **Elimination of Artifact Drifts**: Removes the dependency on external configuration files (JSON/YAML) into the production library.

### Failure Semantics
If a server defined in the Spec is unreachable:
- The Gateway **must still start**.
- The failure is recorded in the Observation layer (JSON).
- The system continues in a degraded state (tools from that server are simply missing from the manifest).

### Test Integration
The **Three-Layer Pattern** enables a robust Tier 3 (Integration) testing strategy:
1.  **Direct Spec Usage**: `GatewayTestClient` imports the `FLEET_SPEC` to obtain the list of clusters and their target slugs.
2.  **Resolution Parity**: The test client uses the production `Resolver` to find the correct local URLs (applying any `.env` or Docker-specific overrides).
3.  **Capability Testing**: Integration tests (e.g., `tests/mcp_servers/gateway/clusters/sanctuary_git/test_gateway.py`) verify that the resolved servers provide the tools defined in their respective cluster specs.
4.  **Mocking Policy**: For unit/logic tests, developers can substitute a "Mock Resolver" that returns local mock SSE servers instead of the real Fleet.

## Requirements
1.  Use the **Spec + Resolver** to determine which servers should be registered and initialized.
2.  Use `gateway_client.py` to register and initialize the resolved servers.
3.  Use `gateway_client.py.get_tools()` to observe tool availability.
4.  Persist the observed results into `fleet_registry.json` as a discovery manifest.
5.  **No production logic may rely on `fleet_registry.json` as an input.**

## Data Structure & Reference Example (`fleet_registry.json`)
The JSON file acts as a **Discovery Manifest**, populated by the `gateway_client.py` after performing handshakes. It follows this hierarchical structure:

- **Top Level**: `fleet_servers` (Object)
- **Key**: Alias (e.g., `utils`, `git`)
- **Properties**: `slug`, `url`, `description`, and a `tools` (Array) containing name, description, and input schema.

### Reference Example

```json
{
  "fleet_servers": {
    "utils": {
      "slug": "sanctuary_utils",
      "url": "http://sanctuary_utils:8000/sse",
      "description": "Calculator, Time, Search",
      "tools": [
        {
          "name": "calculate",
          "description": "Perform math operations",
          "inputSchema": {}
        }
      ]
    },
    "git": {
      "slug": "sanctuary_git",
      "url": "http://sanctuary_git:8000/sse",
      "description": "Protocol 101 Git Operations",
      "tools": []
    }
  }
}
```

## Implementation Strategy

1.  **Define Spec**: Create `fleet_spec.py` with the `FLEET_SPEC` mapping.
2.  **Define Resolver**: Create `fleet_resolver.py` to handle `os.getenv` overrides.
3.  **Slim the Client**: Remove all fleet-specific logic from `gateway_client.py`.
4.  **Create CLI Orchestrator**: Build a separate CLI tool/script that uses the **Resolver** to drive discovery and update the **Observation** manifest.
