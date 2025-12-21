# Gateway Verification Matrix & Operations Tracker

This document tracks the verification status of every operation across the Hybrid Fleet (Local Scripts vs. Docker Containers).
**Legend:** ✅ Verified | ⏳ In Progress | 🔴 Pending | 🚫 Skipped/NA

**Last Updated:** 2025-12-19 (56/56 Tests Passing - 100% Functional Parity)

## 1. sanctuary_utils (Port 8100)
| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `time.get_current_time` | ✅ | ✅ | ✅ | |
| `calculator.add` | ✅ | ✅ | ✅ | |

## 2. sanctuary_filesystem (Port 8101)
| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `code_read` | ✅ | ✅ | ✅ | |
| `code_list_files` | ✅ | ✅ | ✅ | |
| `code_lint` | ✅ | ✅ | ✅ | |
| `code_analyze` | ✅ | ✅ | ✅ | |
| `code_find_file` | ✅ | ✅ | ✅ | |

## 3. sanctuary_network (Port 8102)
| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Tools** | | | | |
| `fetch_url` | ✅ | ✅ | ✅ | |
| `check_site_status` | ✅ | ✅ | ✅ | |

## 4. sanctuary_git (Port 8103)
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
| `git_add` | ✅ | ✅ | ✅ | |
| `git_start_feature` | ✅ | ✅ | ✅ | |
| `git_smart_commit` | ✅ | ✅ | ✅ | |

## 5. sanctuary_cortex (Port 8104)
*Includes: RAG Cortex + Forge LLM*
| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `health` | ✅ | ✅ | ✅ | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **RAG Cortex** | | | | |
| `cortex_get_stats` | ✅ | ✅ | ✅ | |
| `cortex_cache_stats` | ✅ | ✅ | ✅ | |
| `cortex_query` | ✅ | ✅ | ✅ | |
| `cortex_guardian_wakeup` | ✅ | ✅ | ✅ | P114 |
| **Forge LLM** | | | | |
| `query_sanctuary_model` | ✅ | ✅ | ✅ | |
| `check_sanctuary_model_status`| ✅ | ✅ | ✅ | |

## 6. sanctuary_domain (Port 8105)
*Includes: Chronicle, Protocol, Task, ADR, Agent Persona, Config*
| Operation (Tool) | Classic (Local) | Fleet (Docker) | Integration (SSE) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Infra** | | | | |
| `sse_endpoint` | ✅ | ✅ | ✅ | |
| **Chronicle** | | | | |
| `chronicle_list_entries` | ✅ | ✅ | ✅ | |
| `chronicle_search` | ✅ | ✅ | ✅ | |
| `chronicle_create_entry` | ✅ | ✅ | ✅ | |
| **Protocol** | | | | |
| `protocol_list` | ✅ | ✅ | ✅ | |
| `protocol_get` | ✅ | ✅ | ✅ | |
| `protocol_search` | ✅ | ✅ | ✅ | |
| `protocol_create` | ✅ | ✅ | ✅ | |
| **Task** | | | | |
| `list_tasks` | ✅ | ✅ | ✅ | |
| **ADR** | | | | |
| `adr_list` | ✅ | ✅ | ✅ | |
| `adr_search` | ✅ | ✅ | ✅ | |
| **Agent Persona** | | | | |
| `persona_list_roles` | ✅ | ✅ | ✅ | |
| `persona_dispatch` | ✅ | ✅ | ✅ | |
| **Config** | | | | |
| `config_list` | ✅ | ✅ | ✅ | |
| `config_read` | ✅ | ✅ | ✅ | |
| **Dev Tools** | | | | |
| `code_check_tools` | ✅ | ✅ | ✅ | |

## 7. Infrastructure (Backend Services)
| Service | Port | Status | Notes |
| :--- | :---: | :---: | :--- |
| `sanctuary_vector_db` | 8110 | ✅ Running | ChromaDB |
| `sanctuary_ollama_mcp` | 11434 | ✅ Running | Ollama |
