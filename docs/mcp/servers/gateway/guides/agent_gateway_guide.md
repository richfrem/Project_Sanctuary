# Agent Gateway Integration Guide

This guide explains how an AI agent (Gemini/Antigravity) can consume MCP tools via the Sanctuary Gateway.

---

## Quick Start - Verified Working Example

```bash
# Set your token (from .env file)
export MCPGATEWAY_BEARER_TOKEN=$(grep MCPGATEWAY_BEARER_TOKEN .env | cut -d'=' -f2 | tr -d '"')

# Call the hello-world tool
curl -k -s -X POST https://localhost:4444/rpc \
  -H "Authorization: Bearer $MCPGATEWAY_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "hello-world-say-hello",
      "arguments": {"name": "Gemini Agent"}
    },
    "id": 1
  }'

# Expected response:
# {"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"Hello, Gemini Agent!"}],"isError":false},"id":1}
```

---

## Gateway Configuration

| Setting | Value |
|---------|-------|
| **External URL** | `https://localhost:4444` |
| **Container URL** | `http://mcp_gateway:8000` |
| **Auth Header** | `Authorization: Bearer <TOKEN>` |
| **Admin UI** | `https://localhost:4444/admin` |

---

## API Reference

### 1. List Tools
```bash
curl -k -s https://localhost:4444/tools \
  -H "Authorization: Bearer $MCPGATEWAY_BEARER_TOKEN" | jq '.[].name'
```

### 2. Call a Tool (JSON-RPC via /rpc)
```bash
curl -k -s -X POST https://localhost:4444/rpc \
  -H "Authorization: Bearer $MCPGATEWAY_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "<tool-name>",
      "arguments": { ... }
    },
    "id": 1
  }'
```

### 3. List Gateways
```bash
curl -k -s https://localhost:4444/gateways \
  -H "Authorization: Bearer $MCPGATEWAY_BEARER_TOKEN" | jq '.[].name'
```

---

## Available Tools

| Tool Name | Description |
|-----------|-------------|
| `hello-world-say-hello` | Says hello to someone |

*Run `GET /tools` for the current full list.*

---

## Python Integration

There are two ways to interact with the Gateway in Python:

### 1. Minimal (requests/httpx)
Use this if you don't want to add dependencies.

```python
import os
import httpx

# Configuration
GATEWAY = os.getenv("MCP_GATEWAY_URL", "https://localhost:4444")
TOKEN = os.getenv("MCPGATEWAY_BEARER_TOKEN")

def call_tool(name: str, arguments: dict):
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
        "id": 1
    }
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(verify=False, http2=False) as client:
        r = client.post(f"{GATEWAY}/rpc", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()
```

### 2. Canonical Library (`gateway_client.py`)
Use this for robust, type-hinted interactions. Located at `mcp_servers/gateway/gateway_client.py`.

```python
from mcp_servers.gateway.gateway_client import execute_mcp_tool

# Example: Get Git Status
result = execute_mcp_tool(
    tool_name="sanctuary_git-git-get-status",
    arguments={}
)

if result["success"]:
    print(result["result"]["content"][0]["text"])
```

---

## Token Setup

```bash
# Load token from .env
export MCPGATEWAY_BEARER_TOKEN=$(grep MCPGATEWAY_BEARER_TOKEN .env | cut -d'=' -f2 | tr -d '"')
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Invalid token` | Check token is correct RS256 JWT |
| `405 Method Not Allowed` | Use `/rpc` endpoint, not `/tools/call` |
| `503 Service Unavailable` | MCP server unreachable - check container network |
| Tool not found | Use full namespaced name from `GET /tools` |
