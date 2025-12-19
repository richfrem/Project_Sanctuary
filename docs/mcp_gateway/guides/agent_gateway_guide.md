# Agent Gateway Integration Guide

This guide explains how an AI agent (Gemini/Antigravity) can consume MCP tools via the Sanctuary Gateway.

---

## Quick Start - Verified Working Example

```bash
# Set your token (from .env file)
export MCPGATEWAY_BEARER_TOKEN=$(grep MCP_GATEWAY_API_TOKEN .env | cut -d'=' -f2 | tr -d '"')

# Call the hello-world tool
curl -k -s -X POST https://localhost:4444/rpc \
  -H "Authorization: Bearer $MCPGATEWAY_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "hello-world-mcp-say-hello",
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
| **Container URL** | `http://mcpgateway:8000` |
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
| `hello-world-mcp-say-hello` | Says hello to someone |

*Run `GET /tools` for the current full list.*

---

## Python Example

```python
import os
import requests

GATEWAY = "https://localhost:4444"
TOKEN = os.environ.get("MCPGATEWAY_BEARER_TOKEN")

def call_tool(name: str, arguments: dict):
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
        "id": 1
    }
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    r = requests.post(f"{GATEWAY}/rpc", json=payload, headers=headers, verify=False, timeout=30)
    r.raise_for_status()
    return r.json()

# Example
result = call_tool("hello-world-mcp-say-hello", {"name": "Agent"})
print(result["result"]["content"][0]["text"])  # "Hello, Agent!"
```

---

## Token Setup

```bash
# Load token from .env
export MCPGATEWAY_BEARER_TOKEN=$(grep MCP_GATEWAY_API_TOKEN .env | cut -d'=' -f2 | tr -d '"')
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Invalid token` | Check token is correct RS256 JWT |
| `405 Method Not Allowed` | Use `/rpc` endpoint, not `/tools/call` |
| `503 Service Unavailable` | MCP server unreachable - check container network |
| Tool not found | Use full namespaced name from `GET /tools` |
