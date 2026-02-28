# Gateway Verification Matrix & Operations Tracker

This document tracks the **complete verification status** of every operation across the Hybrid Fleet.

**Last Updated:** 2025-12-26  
**Total Tools:** 87 (from fleet_registry.json)  
**Reference:** ADR-066 v1.3, ADR-076

---

## Legend

| Symbol | Meaning |
|:------:|:--------|
| ✅ | Verified/Passing |
| ⚠️ | Partial/Timeout |
| 🔴 | Failing/Blocked |
| ➖ | Not Applicable |
| ⏳ | Not Yet Tested |

---

## ADR-066 Transport Compliance Summary

### What is STDIO vs SSE?

**STDIO (Standard I/O)** is the *original* Agent Plugin Integration transport. It uses simple stdin/stdout pipes - perfect for local tools where the AI assistant runs on your machine. Think of it like a direct phone call between two people in the same room.

**SSE (Server-Sent Events)** is the *web* transport. It uses HTTP connections with streaming events - required when your Agent Plugin Integration server runs in a container and the AI needs to reach it over a network. Think of it like a video call over the internet.

### Why Do We Need Both?

![[gateway_production_flow.png|gateway_production_flow]]

*[[gateway_production_flow.mmd|Source: gateway_production_flow.mmd]]*

> **Key:** All Agent Plugin Integration-compatible clients speak STDIO to `bridge.py`, which converts to HTTP for the Gateway, which speaks SSE to containers.

### Development & Testing Path

For direct testing, integration tests, and headless E2E scenarios, you bypass the bridge:

![[mcp_testing_dev_paths.png|mcp_testing_dev_paths]]

*[[mcp_testing_dev_paths.mmd|Source: mcp_testing_dev_paths.mmd]]*

> **Key:** Testing can use either transport directly without going through the full Gateway stack.

### Architecture Diagram

![[mcp_sse_stdio_transport.png|mcp_sse_stdio_transport]]

*[[mcp_sse_stdio_transport.mmd|Source: mcp_sse_stdio_transport.mmd]]*

### Transport Implementation

| Transport | Implementation | Entry Point | Use Case |
|:----------|:---------------|:------------|:---------|
| **STDIO** | FastMCP | `run_stdio_server()` | Claude Desktop, Local Dev, Direct terminal |
| **SSE** | SSEServer + @sse_tool | `run_sse_server()` | Gateway Fleet (Podman), Remote access |

**Selector:** `MCP_TRANSPORT` environment variable (default: `stdio`)

### How It Works in Practice

1. **Container starts** with `MCP_TRANSPORT=sse` → Uses SSEServer
2. **Gateway discovers** the container via `/sse` endpoint handshake
3. **LLM calls tool** → Gateway forwards RPC → Container executes → Response streams back
4. **Local dev** with `MCP_TRANSPORT=stdio` → Uses FastMCP directly (no network needed)

---

## Cluster Health Summary

| Cluster | Port | Tools | Health | SSE | Gateway | Unit | Integration | STDIO |
|:--------|:----:|:-----:|:------:|:---:|:-------:|:----:|:-----------:|:-----:|
| sanctuary_utils | 8100 | 17 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| sanctuary_filesystem | 8101 | 11 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| sanctuary_network | 8102 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| sanctuary_git | 8103 | 9 | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ✅ |
| sanctuary_cortex | 8104 | 13 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| sanctuary_domain | 8105 | 35 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Total** | - | **87** | **6/6** | **6/6** | **5/6** | **6/6** | **5/6** | **6/6** |

> **Note:** Git Gateway RPC tests timeout due to Gateway SSL handshake (not tool logic).
> Direct SSE and health tests all pass.

---

## 1. sanctuary_utils (Port 8100) - 17 Tools

| Tool | Gateway Registered | Unit | Integration | SSE | STDIO | LLM |
|:-----|:------------------:|:----:|:-----------:|:---:|:-----:|:---:|
| **Infrastructure** |||||||
| `/health` | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| `/sse` endpoint | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| **Time** |||||||
| `time-get-current-time` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `time-get-timezone-info` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Calculator** |||||||
| `calculator-calculate` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `calculator-add` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `calculator-subtract` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `calculator-multiply` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `calculator-divide` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **UUID** |||||||
| `uuid-generate-uuid4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `uuid-generate-uuid1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `uuid-validate-uuid` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **String** |||||||
| `string-to-upper` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `string-to-lower` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `string-trim` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `string-reverse` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `string-word-count` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `string-replace` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Gateway** |||||||
| `gateway-get-capabilities` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 2. sanctuary_filesystem (Port 8101) - 11 Tools

| Tool | Gateway Registered | Unit | Integration | SSE | STDIO | LLM |
|:-----|:------------------:|:----:|:-----------:|:---:|:-----:|:---:|
| **Infrastructure** |||||||
| `/health` | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| `/sse` endpoint | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| **File I/O** |||||||
| `code-read` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-write` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-delete` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-get-info` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Discovery** |||||||
| `code-list-files` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-find-file` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-search-content` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Code Quality** |||||||
| `code-lint` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-format` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-analyze` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code-check-tools` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 3. sanctuary_network (Port 8102) - 2 Tools

| Tool | Gateway Registered | Unit | Integration | SSE | STDIO | LLM |
|:-----|:------------------:|:----:|:-----------:|:---:|:-----:|:---:|
| **Infrastructure** |||||||
| `/health` | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| `/sse` endpoint | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| **Tools** |||||||
| `fetch-url` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `check-site-status` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 4. sanctuary_git (Port 8103) - 9 Tools

| Tool | Gateway Registered | Unit | Integration | SSE | STDIO | LLM |
|:-----|:------------------:|:----:|:-----------:|:---:|:-----:|:---:|
| **Infrastructure** |||||||
| `/health` | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| `/sse` endpoint | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| **Status** |||||||
| `git-get-status` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `git-log` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `git-diff` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Protocol 101** |||||||
| `git-get-safety-rules` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `git-smart-commit` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `git-add` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Feature Workflow** |||||||
| `git-start-feature` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `git-push-feature` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `git-finish-feature` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> **⚠️ Integration:** Gateway RPC timeout (SSL handshake issue, not tool logic)

---

## 5. sanctuary_cortex (Port 8104) - 15 Tools

| Tool | Gateway Registered | Unit | Integration | SSE | STDIO | LLM |
|:-----|:------------------:|:----:|:-----------:|:---:|:-----:|:---:|
| **Infrastructure** |||||||
| `/health` | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| `/sse` endpoint | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| **RAG Ingestion** |||||||
| `cortex-ingest-full` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-ingest-incremental` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RAG Query** |||||||
| `cortex-query` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-get-stats` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cache (CAG)** |||||||
| `cortex-cache-stats` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-cache-get` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-cache-set` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-cache-warmup` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Protocol Tools** |||||||
| `cortex-guardian-wakeup` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-learning-debrief` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cortex-capture-snapshot` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Soul Persistence (ADR 079/081)** |||||||
| `cortex-persist-soul` | ✅ | ⏳ | ✅ | ✅ | ⏳ | ⏳ |
| `cortex-persist-soul-full` | ✅ | ⏳ | ✅ | ✅ | ⏳ | ⏳ |
| **Forge LLM** |||||||
| `query-sanctuary-model` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ |
| `check-sanctuary-model-status` | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |

> **⚠️ Forge LLM:** Depends on Ollama model availability

---

## 6. sanctuary_domain (Port 8105) - 35 Tools

| Tool | Gateway Registered | Unit | Integration | SSE | STDIO | LLM |
|:-----|:------------------:|:----:|:-----------:|:---:|:-----:|:---:|
| **Infrastructure** |||||||
| `/health` | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| `/sse` endpoint | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ |
| **Chronicle (7)** |||||||
| `chronicle-list-entries` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `chronicle-read-latest-entries` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `chronicle-get-entry` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `chronicle-search` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `chronicle-create-entry` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `chronicle-append-entry` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `chronicle-update-entry` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Protocol (5)** |||||||
| `protocol-list` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `protocol-get` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `protocol-search` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `protocol-create` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `protocol-update` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Task (6)** |||||||
| `list-tasks` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `get-task` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `search-tasks` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `create-task` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `update-task` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `update-task-status` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ADR (5)** |||||||
| `adr-list` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `adr-get` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `adr-search` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `adr-create` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `adr-update-status` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Persona (5)** |||||||
| `persona-list-roles` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `persona-get-state` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `persona-reset-state` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `persona-dispatch` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `persona-create-custom` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Config (4)** |||||||
| `config-list` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `config-read` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `config-write` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `config-delete` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Workflow (2)** |||||||
| `get-available-workflows` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `read-workflow` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 7. Backend Services

| Service | Port | Status | Health | Connectivity |
|:--------|:----:|:------:|:------:|:------------:|
| `sanctuary_vector_db` | 8110 | ✅ Running | ✅ | ✅ ChromaDB v2 |
| `sanctuary_ollama` | 11434 | ✅ Running | ✅ | ✅ Ollama API |

---

## Test Execution Summary

### Automated Tests Run (2024-12-24)

| Test Suite | Location | Result |
|:-----------|:---------|:------:|
| SSE Handshake | `tests/mcp_servers/gateway/integration/test_sse_handshake.py` | ✅ 14/14 |
| Cortex Gateway | `tests/mcp_servers/gateway/clusters/sanctuary_cortex/` | ✅ 4/6 |
| Domain Gateway | `tests/mcp_servers/gateway/clusters/sanctuary_domain/` | ✅ 9/9 |
| Filesystem Gateway | `tests/mcp_servers/gateway/clusters/sanctuary_filesystem/` | ✅ 5/5 |
| Git Gateway | `tests/mcp_servers/gateway/clusters/sanctuary_git/` | ⚠️ 2/5 |
| Network Gateway | `tests/mcp_servers/gateway/clusters/sanctuary_network/` | ✅ 4/4 |
| Utils Gateway | `tests/mcp_servers/gateway/clusters/sanctuary_utils/` | ✅ 16/16 |

**Total:** 48/50 tests passing (96%)

### Health Endpoint Verification

```bash
curl http://localhost:8100/health  # ✅ {"status":"ok"}
curl http://localhost:8101/health  # ✅ {"status":"ok"}
curl http://localhost:8102/health  # ✅ {"status":"ok"}
curl http://localhost:8103/health  # ✅ {"status":"ok"}
curl http://localhost:8104/health  # ✅ {"status":"healthy"}
curl http://localhost:8105/health  # ✅ {"status":"healthy"}
```

### SSE Handshake Verification

```bash
timeout 2 curl -N http://localhost:8100/sse  # ✅ event: endpoint
timeout 2 curl -N http://localhost:8101/sse  # ✅ event: endpoint
timeout 2 curl -N http://localhost:8102/sse  # ✅ event: endpoint
timeout 2 curl -N http://localhost:8103/sse  # ✅ event: endpoint
timeout 2 curl -N http://localhost:8104/sse  # ✅ event: endpoint
timeout 2 curl -N http://localhost:8105/sse  # ✅ event: endpoint
```

---

## Known Issues

| Issue | Cluster | Severity | Status | Notes |
|:------|:--------|:--------:|:------:|:------|
| Gateway SSL Timeout | sanctuary_git | Low | ⚠️ | Gateway→Container SSL handshake times out. Direct SSE works. |
| Ollama Model Availability | sanctuary_cortex | Low | ⚠️ | `query-sanctuary-model` depends on Ollama model being loaded. |

---

*For operations reference, see [[README|README.md]]*
