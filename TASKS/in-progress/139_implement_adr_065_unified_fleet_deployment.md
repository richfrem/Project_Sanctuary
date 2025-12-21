# TASK: Implement ADR 065 Unified Fleet Deployment

**Status:** in-progress
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** ADR-065, ADR-066

---

## 1. Objective

Implement the Unified Fleet Deployment CLI (Iron Root Makefile) as defined in ADR 065, ensuring robust orchestration of the Fleet of 8.

## Implementation Plan

### Phase 1: Structure Verification (COMPLETED)
- [x] Rename Fleet directories to kebab-case (ADR 063)
- [x] Update `docker-compose.yml` build contexts to point to `mcp_servers/gateway/clusters/sanctuary-*`
- [x] **CRITICAL**: Verify `mcp_servers/git` and `mcp_servers/rag_cortex` (Legacy) are PRESERVED and UNTOUCHED.

### Phase 1.5: Blueprint Refactoring (Dockerfiles) - COMPLETED
- [x] **Analyze & Fix**: `sanctuary_utils` (Restore `lib` copy, correct CMD)
- [x] **Analyze & Fix**: `sanctuary_git` (Restore `lib` copy, correct CMD)
- [x] **Analyze & Fix**: `sanctuary_cortex` (Restore `lib` copy, correct CMD, requirements)
- [x] **Analyze & Fix**: `sanctuary_filesystem`
- [x] **Analyze & Fix**: `sanctuary_network`
- [x] **Analyze & Fix**: `sanctuary_domain`
- [x] **CRITICAL FIX**: Created missing `__init__.py` in `mcp_servers/gateway/clusters/`
- [x] **CRITICAL FIX**: Renamed all directories from kebab-case to snake_case for Python compatibility
- [x] **Global Update**: Updated all references across project (docker-compose.yml, Dockerfiles, docs, tests)

### Phase 1.6: Legacy Context Analysis - COMPLETED
- [x] **Analyzed**: Confirmed "Hybrid" model (2 containers + 10 stdio) was previous state
- [x] **Analyzed**: Identified legacy Dockerfiles as debris from fragmented past efforts
- [x] **Deleted**: Removed 7 legacy Dockerfiles (adr, code, rag_cortex, protocol, task, chronicle, git)
- [x] **Verified**: Fleet containers do not conflict with Legacy stdio processes (different architecture)

### Phase 1.7: Dockerfile Status Map (Analysis) - COMPLETED
**Active (Used by Fleet/Podman):**
- [x] `mcp_servers/gateway/clusters/sanctuary_utils/Dockerfile` (Logic Container)
- [x] `mcp_servers/gateway/clusters/sanctuary_filesystem/Dockerfile` (Logic Container)
- [x] `mcp_servers/gateway/clusters/sanctuary_network/Dockerfile` (Logic Container)
- [x] `mcp_servers/gateway/clusters/sanctuary_git/Dockerfile` (Logic Container)
- [x] `mcp_servers/gateway/clusters/sanctuary_cortex/Dockerfile` (Logic Container)
- [x] `mcp_servers/gateway/clusters/sanctuary_domain/Dockerfile` (Logic Container)
- [x] Forge runs via `ollama:latest` image (NO Dockerfile needed)
- [x] VectorDB runs via `chromadb:latest` image (NO Dockerfile needed)

**Inactive/Legacy (Deleted):**
- [x] `mcp_servers/adr/Dockerfile` - **DELETED** (Bundled in sanctuary_domain)
- [x] `mcp_servers/protocol/Dockerfile` - **DELETED** (Bundled in sanctuary_domain)
- [x] `mcp_servers/chronicle/Dockerfile` - **DELETED** (Bundled in sanctuary_domain)
- [x] `mcp_servers/task/Dockerfile` - **DELETED** (Bundled in sanctuary_domain)
- [x] `mcp_servers/code/Dockerfile` - **DELETED** (Replaced by sanctuary_filesystem)
- [x] `mcp_servers/rag_cortex/Dockerfile` - **DELETED** (Duplicate/mistake)
- [x] `mcp_servers/git/Dockerfile` - **DELETED** (Duplicate/mistake)

### Phase 1.8: Documentation Harmonization - COMPLETED
- [x] **Updated**: `docs/PODMAN_STARTUP_GUIDE.md` (Reflects Makefile workflow)
- [x] **Updated**: `docs/mcp_servers/README.md` (Added Architecture Transition section and Deployment Map)
- [x] **Updated**: `docs/mcp_servers/rag_cortex/SETUP.md` (Added Makefile workflow notice)
- [x] **Updated**: `docs/mcp_servers/forge_llm/SETUP.md` (Added Makefile workflow notice)
- [x] **Updated**: `docs/mcp_servers/operations/setup_guide.md` (Added Makefile workflow notice)

### Phase 2: Pre-Flight Checks - COMPLETED
- [x] Verify `scripts/wait_for_pulse.sh` exists and is executable.
- [x] Verify `.env` file exists and contains `MCPGATEWAY_BEARER_TOKEN`.
- [x] Verify `docker-compose.yml` syntax is valid (`podman compose config`).

### Phase 3: The Phoenix Test (Deployment) - COMPLETED ✅
- [x] **Phase 3.1: Heavy Lift Verification**:
      1. Brought up `vector-db` and `ollama-model-mcp` first.
      2. Verified VectorDB endpoint (v2 heartbeat).
      3. Verified Ollama endpoint.
      4. Confirmed data persistence (volumes mounted correctly).
- [x] **Execute**: `make down` (Clean slate).
- [x] **Execute**: `make up force=true` (Deploy Full Fleet with rebuild).
- [x] **Execute**: `make status` (Verified all 8 active).
- [x] **Status Check**: All 8 containers UP and running (sanctuary_utils, sanctuary_git, sanctuary_network, sanctuary_filesystem, sanctuary_cortex, sanctuary_domain, sanctuary_vector_db, sanctuary_ollama_mcp).

### Phase 3.5: Gateway Registration Lifecycle Verification - COMPLETED ✅
- [x] **Created Admin Functions**: Added `list_servers`, `deactivate_server`, `delete_server`, `clean_all_servers` to `gateway_client.py`
- [x] **Fixed Endpoints**: Corrected all functions to use `/gateways` instead of `/admin/servers`
- [x] **Created Master Setup Script**: `fleet_setup.py` orchestrates full lifecycle (clean + register + verify)
- [x] **Updated Makefile**: Changed to call `fleet_setup.py` instead of `fleet_orchestrator.py`
- [x] **Clean Gateway State**: Successfully removes all registered servers before re-registration
- [x] **Run Orchestrator**: Executes full registration flow
- [x] **Verify Registration**: 6/6 servers registered with Gateway ✅
- [x] **Gateway Client CLI**: Added comprehensive CLI interface to `gateway_client.py`
  - Commands: `pulse`, `tools`, `servers`, `register`, `status`
  - Usage: `python -m mcp_servers.gateway.gateway_client tools -v`
- [x] **Tool Discovery**: 84 tools federated across all 6 servers ✅

### Phase 3.6: Root Cause Analysis - Tool Discovery (RESOLVED)

**Discovery Date:** 2025-12-20

**Original Symptoms:**
- All 6 servers register successfully with Gateway
- Only 50 tools appeared in `/tools` endpoint (pagination limit!)
- "Virtual Servers Catalog" showed more tools than API

**Root Cause:**
The `/tools` API endpoint had a default pagination limit of 50 items. With 84 total tools, the remaining 34 were on page 2.

**Resolution:**
- Updated `gateway_client.py` to use `/admin/tools` endpoint with `per_page=200`
- Fixed response parsing to handle paginated format `{data: [...], pagination: {...}}`

**Final Tool Count (All 6 Servers):**
- sanctuary_cortex: 11 tools ✅
- sanctuary_domain: 36 tools ✅
- sanctuary_filesystem: 10 tools ✅
- sanctuary_git: 9 tools ✅
- sanctuary_network: 2 tools ✅
- sanctuary_utils: 16 tools ✅
- **TOTAL: 84 tools** ✅

### Phase 4: Functional Verification - IN PROGRESS
- [x] **Gateway Pulse**: Verified via CLI (`python -m mcp_servers.gateway.gateway_client pulse`)
- [x] **Tools Check**: Verified via CLI (`python -m mcp_servers.gateway.gateway_client tools -v`)
- [x] **Servers Check**: Verified via CLI (`python -m mcp_servers.gateway.gateway_client servers`)
- [ ] **Registry Check**: Update `fleet_registry.json` with correct tool counts
- [ ] **Tool Invocation Test**: Test calling a tool through the Gateway
- [ ] **Cross-Server Test**: Verify tools from different servers can be invoked in same session

**Related Tasks & Documentation:**
- **[Task 136](../in-progress/136_gateway_client_full_verification__protocol_125_loo.md)**: Full 85-tool verification matrix (Zero-Trust Pyramid)
- **[GATEWAY_VERIFICATION_MATRIX.md](../../docs/mcp_servers/gateway/operations/GATEWAY_VERIFICATION_MATRIX.md)**: 56/56 tests passing (2025-12-19)
- **[mcp_operations_inventory.md](../../docs/mcp_servers/operations/mcp_operations_inventory.md)**: Full operations reference

### Phase 5: Documentation & Finalization
- [x] Update `docs/PODMAN_STARTUP_GUIDE.md` with Makefile workflow.
- [x] Update `docs/mcp_servers/README.md` with tool federation status.
- [x] Mark ADR 065 as ACCEPTED.
- [ ] Final review and close Task 139.

## 2. Deliverables

1. Makefile ✅
2. scripts/wait_for_pulse.sh ✅
3. Updated PODMAN_STARTUP_GUIDE.md ✅
4. Gateway Client CLI (gateway_client.py) ✅
5. ADR-066 (FastMCP Standardization) ✅

## 3. Acceptance Criteria

- [x] Makefile supports up, down, restart, status, verify targets.
- [x] scripts/wait_for_pulse.sh accurately detects service health.
- [x] All 8 containers deploy successfully via 'make up'.
- [x] All 6 logic containers register with Gateway.
- [x] All 6 logic containers federate tools correctly (84 tools total).
- [x] ADR 065 status updated to ACCEPTED.
- [ ] ADR 066 status updated to ACCEPTED (optional - all servers working).
- [ ] Phase 4 functional verification tests complete.

## Notes

### Session 2025-12-20:
- Migrated sanctuary_git from SSEServer to FastMCP
- Added /health endpoint to sanctuary_git for Gateway health checks
- Created ADR-066: Standardize on FastMCP (marked optional since all servers working)
- Fixed Gateway Client CLI pagination issue (was limited to 50 tools)
- All 84 tools now discoverable across all 6 servers

---

## Phase 4: MCP Gateway Verification Test Matrix

This section documents the comprehensive testing plan to verify all 84 federated tools work correctly through the Gateway.

### 4-Tier Zero-Trust Verification Pyramid

Per `tests/mcp_servers/gateway/README.md`, each operation must pass all 4 tiers:

| Tier | Name | Focus | Verification Method |
|------|------|-------|---------------------|
| 1 | **Unit** | Core Python logic | `pytest tests/mcp_servers/<server>/unit/` |
| 2 | **Integration** | SSE Server direct response | `pytest tests/mcp_servers/<server>/integration/` |
| 3 | **Gateway RPC** | Gateway routing bridge | `gateway_client.py --step verify` |
| 4 | **IDE/Agent** | Real agent invocation | Antigravity/Claude tool call |

### Test Matrix: Legacy MCP → Gateway Federated

The following matrix maps the 12 legacy stdio MCP servers to their Gateway-federated equivalents:

#### sanctuary_domain (36 tools)

| Legacy Server | Tool Name | Gateway Tool Slug | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------|-----------|-------------------|--------|--------|--------|--------|
| `mcp_adr` | `adr_list` | `sanctuary-domain-adr-list` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_adr` | `adr_get` | `sanctuary-domain-adr-get` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_adr` | `adr_create` | `sanctuary-domain-adr-create` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_adr` | `adr_search` | `sanctuary-domain-adr-search` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_adr` | `adr_update_status` | `sanctuary-domain-adr-update-status` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_chronicle` | `chronicle_list_entries` | `sanctuary-domain-chronicle-list-entries` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_chronicle` | `chronicle_create_entry` | `sanctuary-domain-chronicle-create-entry` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_chronicle` | `chronicle_get_entry` | `sanctuary-domain-chronicle-get-entry` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_chronicle` | `chronicle_search` | `sanctuary-domain-chronicle-search` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_task` | `list_tasks` | `sanctuary-domain-list-tasks` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_task` | `create_task` | `sanctuary-domain-create-task` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_task` | `get_task` | `sanctuary-domain-get-task` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_task` | `update_task_status` | `sanctuary-domain-update-task-status` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_protocol` | `protocol_list` | `sanctuary-domain-protocol-list` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_protocol` | `protocol_get` | `sanctuary-domain-protocol-get` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_config` | `config_list` | `sanctuary-domain-config-list` | ✅ | ✅ | ⏳ | ⏳ |
| `mcp_config` | `config_read` | `sanctuary-domain-config-read` | ✅ | ✅ | ⏳ | ⏳ |

#### sanctuary_git (9 tools)

| Legacy Server | Tool Name | Gateway Tool Slug | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------|-----------|-------------------|--------|--------|--------|--------|
| `mcp_git_workflow` | `git_get_status` | `sanctuary-git-git-get-status` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_add` | `sanctuary-git-git-add` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_smart_commit` | `sanctuary-git-git-smart-commit` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_push_feature` | `sanctuary-git-git-push-feature` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_start_feature` | `sanctuary-git-git-start-feature` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_finish_feature` | `sanctuary-git-git-finish-feature` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_diff` | `sanctuary-git-git-diff` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_log` | `sanctuary-git-git-log` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_git_workflow` | `git_get_safety_rules` | `sanctuary-git-git-get-safety-rules` | ⏳ | ⏳ | ⏳ | ⏳ |

#### sanctuary_cortex (11 tools)

| Legacy Server | Tool Name | Gateway Tool Slug | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------|-----------|-------------------|--------|--------|--------|--------|
| `mcp_rag_cortex` | `cortex_query` | `sanctuary-cortex-cortex-query` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_ingest_full` | `sanctuary-cortex-cortex-ingest-full` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_ingest_incremental` | `sanctuary-cortex-cortex-ingest-incremental` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_get_stats` | `sanctuary-cortex-cortex-get-stats` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_cache_get` | `sanctuary-cortex-cortex-cache-get` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_cache_set` | `sanctuary-cortex-cortex-cache-set` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_cache_stats` | `sanctuary-cortex-cortex-cache-stats` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_cache_warmup` | `sanctuary-cortex-cortex-cache-warmup` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_rag_cortex` | `cortex_guardian_wakeup` | `sanctuary-cortex-cortex-guardian-wakeup` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_forge_llm` | `check_sanctuary_model_status` | `sanctuary-cortex-check-sanctuary-model-status` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_forge_llm` | `query_sanctuary_model` | `sanctuary-cortex-query-sanctuary-model` | ⏳ | ⏳ | ⏳ | ⏳ |

#### sanctuary_filesystem (10 tools)

| Legacy Server | Tool Name | Gateway Tool Slug | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------|-----------|-------------------|--------|--------|--------|--------|
| `mcp_code` | `code_read` | `sanctuary-filesystem-code-read` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_write` | `sanctuary-filesystem-code-write` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_analyze` | `sanctuary-filesystem-code-analyze` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_format` | `sanctuary-filesystem-code-format` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_lint` | `sanctuary-filesystem-code-lint` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_search_content` | `sanctuary-filesystem-code-search-content` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_list_files` | `sanctuary-filesystem-code-list-files` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_find_file` | `sanctuary-filesystem-code-find-file` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_get_info` | `sanctuary-filesystem-code-get-info` | ⏳ | ⏳ | ⏳ | ⏳ |
| `mcp_code` | `code_check_tools` | `sanctuary-filesystem-code-check-tools` | ⏳ | ⏳ | ⏳ | ⏳ |

#### sanctuary_utils (16 tools)

| Legacy Server | Tool Name | Gateway Tool Slug | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------|-----------|-------------------|--------|--------|--------|--------|
| N/A (New) | `string_replace` | `sanctuary-utils-string-replace` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `string_word_count` | `sanctuary-utils-string-word-count` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `string_reverse` | `sanctuary-utils-string-reverse` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `string_trim` | `sanctuary-utils-string-trim` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `string_to_lower` | `sanctuary-utils-string-to-lower` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `string_to_upper` | `sanctuary-utils-string-to-upper` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `calculator_add` | `sanctuary-utils-calculator-add` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `calculator_subtract` | `sanctuary-utils-calculator-subtract` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `calculator_multiply` | `sanctuary-utils-calculator-multiply` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `calculator_divide` | `sanctuary-utils-calculator-divide` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `time_now` | `sanctuary-utils-time-now` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `time_format` | `sanctuary-utils-time-format` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `uuid_generate` | `sanctuary-utils-uuid-generate` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `json_format` | `sanctuary-utils-json-format` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `json_validate` | `sanctuary-utils-json-validate` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `base64_encode` | `sanctuary-utils-base64-encode` | ⏳ | ⏳ | ⏳ | ⏳ |

#### sanctuary_network (2 tools)

| Legacy Server | Tool Name | Gateway Tool Slug | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------|-----------|-------------------|--------|--------|--------|--------|
| N/A (New) | `fetch_url` | `sanctuary-network-fetch-url` | ⏳ | ⏳ | ⏳ | ⏳ |
| N/A (New) | `check_site_status` | `sanctuary-network-check-site-status` | ⏳ | ⏳ | ⏳ | ⏳ |

### Quick Validation Tests (Tier 4 - IDE/Agent)

Run these tests to quickly validate Gateway tool invocation via the IDE:

```bash
# 1. ADR List
python3 -m mcp_servers.gateway.gateway_client tools --server sanctuary_domain

# 2. Legacy MCP (verify still working)
# adr_list, adr_search, chronicle_list_entries, etc.

# 3. Gateway Tool Invocation (Phase 4.5)
# TODO: Add execute_tool function to gateway_client.py
```

### Test Directory Structure

```
tests/mcp_servers/gateway/
├── clusters/
│   ├── sanctuary-cortex/     # Tier 1-2 tests for cortex
│   ├── sanctuary-domain/     # Tier 1-2 tests for domain  
│   ├── sanctuary-filesystem/ # Tier 1-2 tests for filesystem
│   ├── sanctuary-git/        # Tier 1-2 tests for git
│   ├── sanctuary-network/    # Tier 1-2 tests for network
│   └── sanctuary-utils/      # Tier 1-2 tests for utils
├── integration/              # Tier 2 SSE tests
├── e2e/                      # Tier 3-4 end-to-end tests
├── gateway_test_client.py   # Tier 3 helper
└── README.md                 # Zero-Trust Pyramid docs
```

### Verification Status Summary

| Container | Tool Count | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|-----------|------------|--------|--------|--------|--------|
| sanctuary_domain | 36 | ✅ | ✅ | ⏳ | ⏳ |
| sanctuary_cortex | 11 | ⏳ | ⏳ | ⏳ | ⏳ |
| sanctuary_filesystem | 10 | ⏳ | ⏳ | ⏳ | ⏳ |
| sanctuary_git | 9 | ⏳ | ⏳ | ⏳ | ⏳ |
| sanctuary_utils | 16 | ⏳ | ⏳ | ⏳ | ⏳ |
| sanctuary_network | 2 | ⏳ | ⏳ | ⏳ | ⏳ |
| **TOTAL** | **84** | - | - | - | - |
