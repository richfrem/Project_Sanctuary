# Cluster: sanctuary_domain

**Role:** High-level business logic, multi-agent coordination, and strategic mission management.  
**Port:** 8105  
**Front-end Cluster:** ✅ Yes

## Overview
The `sanctuary_domain` cluster is the primary reasoning engine for Project Sanctuary. It integrates the foundational operational frameworks (ADR, Chronicle, Protocol, Task) with the strategic orchestration layers of the **Council** and **Orchestrator**.

## Verification Specs (Tier 3: Bridge)
*   **Target:** Gateway Bridge & RPC Routing
*   **Method:** `pytest tests/mcp_servers/gateway/clusters/domain/test_gateway.py`
*   **Auth:** Bearer Token required via `MCPGATEWAY_BEARER_TOKEN`.

## Tool Inventory & Legacy Mapping
| Category | Gateway Tool Name | Legacy Operation | T1/T2 Method |
| :--- | :--- | :--- | :--- |
| **Orchestration** | `sanctuary_domain-orchestrate-mission` | `orchestrator_dispatch_mission` | `pytest tests/mcp_servers/orchestrator/` |
| | `sanctuary_domain-run-strategic-cycle` | `orchestrator_run_strategic_cycle` | |
| **Deliberation**| `sanctuary_domain-council-deliberate` | `council_dispatch` | `pytest tests/mcp_servers/council/` |
| | `sanctuary_domain-council-list-agents`| `council_list_agents` | |
| **ADR** | `sanctuary_domain-adr-create` | `adr_create` | `pytest tests/mcp_servers/adr/` |
| | `sanctuary_domain-adr-update-status`| `adr_update_status` | |
| | `sanctuary_domain-adr-get` | `adr_get` | |
| | `sanctuary_domain-adr-list` | `adr_list` | |
| | `sanctuary_domain-adr-search` | `adr_search` | |
| **Task** | `sanctuary_domain-create-task` | `create_task` | `pytest tests/mcp_servers/task/` |
| | `sanctuary_domain-update-task` | `update_task` | |
| | `sanctuary_domain-update-task-status`| `update_task_status` | |
| | `sanctuary_domain-get-task` | `get_task` | |
| | `sanctuary_domain-list-tasks` | `list_tasks` | |
| | `sanctuary_domain-search-tasks` | `search_tasks` | |
| **Protocol** | `sanctuary_domain-protocol-create` | `protocol_create` | `pytest tests/mcp_servers/protocol/` |
| | `sanctuary_domain-protocol-update` | `protocol_update` | |
| | `sanctuary_domain-protocol-get` | `protocol_get` | |
| | `sanctuary_domain-protocol-list` | `protocol_list` | |
| | `sanctuary_domain-protocol-search` | `protocol_search` | |
| **Chronicle** | `sanctuary_domain-chronicle-create-entry`| `chronicle_create_entry`| `pytest tests/mcp_servers/chronicle/` |
| | `sanctuary_domain-chronicle-append-entry`| `chronicle_append_entry`| |
| | `sanctuary_domain-chronicle-update-entry`| `chronicle_update_entry`| |
| | `sanctuary_domain-chronicle-get-entry`| `chronicle_get_entry` | |
| | `sanctuary_domain-chronicle-list-entries`| `chronicle_list_entries` | |
| | `sanctuary_domain-chronicle-search`| `chronicle_search` | |
| **Persona** | `sanctuary_domain-persona-dispatch` | `persona_dispatch` | `pytest tests/mcp_servers/agent_persona/` |
| | `sanctuary_domain-persona-list-roles` | `persona_list_roles` | |
| | `sanctuary_domain-persona-get-state` | `persona_get_state` | |
| | `sanctuary_domain-persona-reset-state` | `persona_reset_state` | |
| | `sanctuary_domain-persona-create-custom`| `persona_create_custom` | |
| **Config** | `sanctuary_domain-config-list` | `config_list` | `pytest tests/mcp_servers/config/` |
| | `sanctuary_domain-config-read` | `config_read` | |
| | `sanctuary_domain-config-write` | `config_write` | |
| | `sanctuary_domain-config-delete` | `config_delete` | |

## Operational Status
- **Physical Boundary**: This documentation covers all logic running on Port 8105.
- **Inter-Cluster Dependencies**: Calls `sanctuary_utils` for primitives and `sanctuary_filesystem` for workspace mutations.
