# Sanctuary Gateway Module
Location: `mcp_servers/gateway/`

## Overview
This module contains the primary client and management interfaces for the Sanctuary Gateway. It is responsible for server registration, tool discovery, and high-level fleet orchestration using the **3-Layer Declarative Fleet Pattern**.

## Prerequisites
Before interacting with the Gateway, ensure the following infrastructure is operational:
- **Podman**: The "Fleet of 8" runs in isolated containers via Podman.
- **Docker Compose**: Managed via [docker-compose.yml](../../docker-compose.yml).
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

## Documentation
- **[Gateway Architecture](../../docs/mcp_servers/gateway/architecture/ARCHITECTURE.md)**: Deep dive into the 3-layer pattern.
- **[ADR 064](../../ADRs/064_centralized_registry_for_fleet_of_8_mcp_servers.md)**: Design decision for centralized registry.
- **[Podman Startup Guide](../../docs/PODMAN_STARTUP_GUIDE.md)**: Detailed container management instructions.
