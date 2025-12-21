# sanctuary_utils

**Fleet of 7 - Container #1: Low-risk utility tools**

Part of the [Fleet of 7 Architecture](../../ADRs/060_gateway_integration_patterns.md) for containerized MCP servers.

## Overview

A FastAPI-based MCP server providing 16 low-risk utility tools:

| Tool Category | Functions | Count |
|---------------|-----------|-------|
| **Time** | get_current_time, get_timezone_info | 2 |
| **Calculator** | calculate, add, subtract, multiply, divide | 5 |
| **UUID** | generate_uuid4, generate_uuid1, validate_uuid | 3 |
| **String** | to_upper, to_lower, trim, reverse, word_count, replace | 6 |

## Quick Start

```bash
# Start via docker-compose
podman compose up -d sanctuary_utils

# Verify health
curl http://localhost:8100/health

# Call a tool
curl -X POST http://localhost:8100/tools/time.get_current_time -H "Content-Type: application/json" -d '{}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SANCTUARY_UTILS_PORT` | 8100 | External port mapping |
| `MCP_GATEWAY_URL` | https://localhost:4444 | Gateway registration URL |
| `MCPGATEWAY_BEARER_TOKEN` | - | Gateway authentication token |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Tool manifest for Gateway |
| `/sse` | GET | SSE endpoint for MCP Gateway |
| `/tools/{name}` | POST | Direct tool invocation |

## ADR 060 Guardrails Implemented

- ✅ **Guardrail 1:** Fault Containment (per-tool try/except)
- ✅ **Guardrail 2:** Self-Registration (auto-register with Gateway)
- ✅ **Guardrail 3:** Network Addressing (Docker network aliases)
- ✅ **Guardrail 4:** Resource Caps (256M memory, 0.5 CPU)

## Development

```bash
# Hot reload (already configured in docker-compose)
# Edit files in mcp_servers/utils/ - changes reflect automatically

# Run tests
pytest tests/mcp_servers/utils/
```
