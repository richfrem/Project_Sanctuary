# MCP Server Operations Inventory

**Objective**: Comprehensive audit of all `mcp_servers/*/operations.py` modules to document available capabilities and their CLI mapping status in `tools/cli.py`.

**Date**: 2026-02-01
**Related Spec**: 0005-refactor-domain-cli

---

## 1. Summary

| MCP Server | Operations Class | CLI Status | Primary Use |
| :--- | :--- | :--- | :--- |
| `learning` | `LearningOperations` | ✅ Complete | Protocol 128 Learning Loop |
| `rag_cortex` | `CortexOperations` | ✅ Complete | RAG & Vector DB |
| `evolution` | `EvolutionOperations` | ✅ Complete | Fitness Metrics |
| `chronicle` | `ChronicleOperations` | ✅ Complete | Journal Entries |
| `task` | `TaskOperations` | ✅ Complete | Task Management |
| `adr` | `ADROperations` | ✅ Complete | Architecture Decisions |
| `protocol` | `ProtocolOperations` | ✅ Complete | Protocol Documents |
| `forge_llm` | `ForgeOperations` | ✅ Complete | Fine-Tuned Model |
| `git` | `GitOperations` | ⏸️ Out of Scope | Version Control (internal use) |
| `code` | `CodeOperations` | ⏸️ Out of Scope | Code Analysis (native tools sufficient) |
| `orchestrator` | `OrchestratorOperations` | ⏸️ Out of Scope | Task Orchestration (programmatic use) |
| `council` | `CouncilOperations` | ⏸️ Out of Scope | Multi-Agent Deliberation (advanced) |

---

## 2. Fully Mapped Servers

### 2.1 Learning Operations (`mcp_servers/learning/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `learning_debrief(hours)` | `debrief --hours N` | Protocol 128 Phase I |
| `capture_snapshot(type, manifest, context)` | `snapshot --type X` | Protocol 128 Phase V |
| `guardian_wakeup(mode)` | `guardian wakeup --mode X` | Boot Digest |
| `guardian_snapshot(context)` | `guardian snapshot --context X` | Session Pack |
| `persist_soul(request)` | `persist-soul --snapshot X` | Protocol 128 Phase VI |
| `persist_soul_full()` | `persist-soul-full` | ADR 081 Full Sync |
| `_rlm_map(files)` | `rlm-distill <target>` | RLM Distillation |

### 2.2 RAG Cortex Operations (`mcp_servers/rag_cortex/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `ingest_full(purge, dirs)` | `ingest --no-purge --dirs X` | Full ingestion |
| `ingest_incremental(file_paths)` | `ingest --incremental --hours N` | Incremental |
| `query(query, max_results, use_cache)` | `query "text" --max-results N` | Semantic search |
| `get_stats(include_samples)` | `stats --samples` | Health metrics |
| `get_cache_stats()` | `cache-stats` | Cache efficiency |
| `cache_warmup(queries)` | `cache-warmup --queries X` | Pre-populate cache |

### 2.3 Evolution Operations (`mcp_servers/evolution/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `calculate_fitness(content)` | `evolution fitness --file X` | Fitness vector |
| `measure_depth(content)` | `evolution depth "text"` | Technical depth |
| `measure_scope(content)` | `evolution scope "text"` | Architectural scope |

### 2.4 Chronicle Operations (`mcp_servers/chronicle/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `list_entries(limit)` | `chronicle list --limit N` | List entries |
| `search_entries(query)` | `chronicle search "query"` | Search entries |
| `get_entry(number)` | `chronicle get N` | Get specific entry |
| `create_entry(title, content, ...)` | `chronicle create "title" --content X` | Create entry |
| `update_entry(number, updates, reason)` | `chronicle update N --reason X` | ✅ Mapped |

### 2.5 Task Operations (`mcp_servers/task/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `list_tasks(status, priority)` | `task list --status X` | List tasks |
| `get_task(number)` | `task get N` | Get specific task |
| `create_task(title, objective, ...)` | `task create "title" --objective X` | Create task |
| `update_task_status(number, status, notes)` | `task update-status N status --notes X` | Status transition |
| `update_task(number, updates)` | `task update N --field value` | ✅ Mapped |
| `search_tasks(query)` | `task search "query"` | ✅ Mapped |

### 2.6 ADR Operations (`mcp_servers/adr/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `list_adrs(status)` | `adr list --status X` | List ADRs |
| `search_adrs(query)` | `adr search "query"` | Search ADRs |
| `get_adr(number)` | `adr get N` | Get specific ADR |
| `create_adr(title, context, decision, ...)` | `adr create "title" --context X` | Create ADR |
| `update_adr_status(number, status, reason)` | `adr update-status N status --reason X` | ✅ Mapped |

### 2.7 Protocol Operations (`mcp_servers/protocol/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `list_protocols(status)` | `protocol list --status X` | List protocols |
| `search_protocols(query)` | `protocol search "query"` | Search protocols |
| `get_protocol(number)` | `protocol get N` | Get specific protocol |
| `create_protocol(title, content, ...)` | `protocol create "title" --content X` | Create protocol |
| `update_protocol(number, updates, reason)` | `protocol update N --reason X` | ✅ Mapped |

### 2.8 Forge LLM Operations (`mcp_servers/forge_llm/operations.py`)

| Method | CLI Command | Notes |
| :--- | :--- | :--- |
| `query_sanctuary_model(prompt, ...)` | `forge query "prompt"` | ✅ Mapped |
| `check_model_availability()` | `forge status` | ✅ Mapped |

> **Note**: Forge commands require `ollama` package installed.

---

## 3. Out of Scope Servers

### 3.1 Git Operations (`mcp_servers/git/operations.py`)

**Reason**: Used internally by `workflow end`. Native `git` CLI is preferred for direct use.

### 3.2 Code Operations (`mcp_servers/code/operations.py`)

**Reason**: Native tools (`ruff`, `black`, `fd`, `rg`) are more powerful and already available.

### 3.3 Orchestrator Operations (`mcp_servers/orchestrator/operations.py`)

**Reason**: Advanced orchestration designed for programmatic agent use, not CLI.

### 3.4 Council Operations (`mcp_servers/council/operations.py`)

**Reason**: Multi-agent deliberation layer, not suitable for simple CLI exposure.

---

## 4. Conclusion

**Coverage**: 8 of 12 MCP servers are fully mapped to CLI commands.

**All domain entity operations** (Chronicle, Task, ADR, Protocol) are now 100% covered including:
- CRUD operations (list, get, create)
- Update operations (update, update-status)
- Search operations

**Out of scope servers** are either:
- Internal/programmatic (Git, Orchestrator, Council)
- Overlap with native tools (Code)

**No gaps remaining.** All requested operations are implemented.
