# Cluster: sanctuary_cortex

**Role:** Strategic memory, RAG synchronization, and specialized LLM inference.  
**Port:** 8104  
**Front-end Cluster:** ✅ Yes

## Overview
The `sanctuary_cortex` cluster provides the neural foundation for Project Sanctuary. It integrates the **Cortex** (RAG/Memory) layer with the **Forge-LLM** (Fine-tuned inference) layer to provide a unified intelligence gateway.

## Verification Specs (Tier 3: Bridge)
*   **Target:** Gateway Bridge & RPC Routing
*   **Method:** `pytest tests/mcp_servers/gateway/clusters/cortex/test_gateway.py`
*   **Backends**: Interfaces with `sanctuary_vector_db` (:8110) and `sanctuary_ollama` (:11434).

## Tool Inventory & Legacy Mapping
| Category | Gateway Tool Name | Legacy Operation | T1/T2 Method |
| :--- | :--- | :--- | :--- |
| **Inference** | `sanctuary_cortex-query-model` | `query_sanctuary_model` | `pytest tests/mcp_servers/forge_llm/` |
| | `sanctuary_cortex-check-model-status` | `check_model_status` | |
| **Memory** | `sanctuary_cortex-cortex-query` | `cortex_query` | `pytest tests/mcp_servers/rag_cortex/` |
| | `sanctuary_cortex-cortex-ingest-full` | `cortex_ingest_full` | |
| | `sanctuary_cortex-cortex-ingest-incr`| `cortex_ingest_incremental`| |
| | `sanctuary_cortex-cortex-get-stats` | `cortex_get_stats` | |
| **Caching** | `sanctuary_cortex-cortex-cache-get` | `cortex_cache_get` | |
| | `sanctuary_cortex-cortex-cache-set` | `cortex_cache_set` | |
| | `sanctuary_cortex-cortex-cache-stats`| `cortex_cache_stats` | |
| | `sanctuary_cortex-cortex-cache-warmup`| `cortex_cache_warmup` | |
| **Lifecycle** | `sanctuary_cortex-cortex-wakeup` | `cortex_guardian_wakeup` | |

## Operational Status
- **Physical Boundary**: This documentation covers all logic running on Port 8104.
- **Memory Coherence**: Enforces Protocol 102 (Mnemonic Synchronization).
