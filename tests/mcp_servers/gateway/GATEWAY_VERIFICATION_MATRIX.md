# Gateway Verification Matrix & Operations Tracker

This document tracks the verification status of every operation across the Hybrid Fleet (Local Scripts vs. Docker Containers).
**Legend:** ✅ Verified | ⏳ In Progress | 🔴 Pending | 🚫 Skipped/NA

**Last Updated:** 2025-12-19 (39/39 Tests Passing)

## 1. sanctuary-utils (Port 8100)
*Reference: Utils MCP (Pilot)*

| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | Verified in test suite |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `time.get_current_time` | ✅ | ✅ | ✅ | |
| `calculator.add` | ✅ | ✅ | ✅ | |

## 2. sanctuary-filesystem (Port 8101)
*Reference: Code MCP (Section 12 of Inventory)*

| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `code_read` | ✅ | ✅ | ✅ | |
| `code_list_files` | ✅ | ✅ | ✅ | |

## 3. sanctuary-network (Port 8102)
*Reference: Network MCP*

| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `fetch_url` | ✅ | ✅ | ✅ | |
| `check_site_status` | ✅ | ✅ | ✅ | |

## 4. sanctuary-git (Port 8103)
*Reference: Git MCP (Section 5 of Inventory)*

| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `git_get_status` | ✅ | ✅ | ✅ | |
| `git_log` | ✅ | ✅ | ✅ | |
| `git_diff` | ✅ | ✅ | ✅ | |
| `git_get_safety_rules` | ✅ | ✅ | ✅ | |

## 5. sanctuary-cortex (Port 8104)
*Reference: RAG Cortex MCP (Section 6 of Inventory)*

| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `cortex_get_stats` | ✅ | ✅ | ✅ | |
| `cortex_cache_stats` | ✅ | ✅ | ✅ | |
| `cortex_query` | ✅ | ✅ | ✅ | |
| `cortex_guardian_wakeup` | ✅ | ✅ | ✅ | P114 |

## 6. sanctuary-domain (Port 8105)
*Reference: Domain Logic Cluster (Chronicle, Protocol, Task, ADR)*

| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `sse_endpoint` | ✅ | ✅ | ✅ | FastMCP - no /health |
| **Chronicle** | | | | |
| `chronicle_list_entries` | ✅ | ✅ | ✅ | |
| `chronicle_search` | ✅ | ✅ | ✅ | |
| **Protocol** | | | | |
| `protocol_list` | ✅ | ✅ | ✅ | |
| `protocol_get` | ✅ | ✅ | ✅ | |
| **Task** | | | | |
| `list_tasks` | ✅ | ✅ | ✅ | |
| **ADR** | | | | |
| `adr_list` | ✅ | ✅ | ✅ | |
| **Dev Tools** | | | | |
| `code_check_tools` | ✅ | ✅ | ✅ | |

## 7. Infrastructure (Backend Services)

| Service | Port | Status |
|---------|------|--------|
| `sanctuary-vector-db` | 8000 | ✅ Running |
| `sanctuary-ollama-mcp` | 11434 | ✅ Running |
