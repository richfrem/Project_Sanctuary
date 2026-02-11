# CLI Gap Analysis: domain_cli.py vs tools/cli.py

**Objective**: Audit functionality in `scripts/domain_cli.py` (Legacy) against `tools/cli.py` (Target) to ensure full parity for domain operations (Chronicle, Task, ADR, Protocol).

**Strategy**: `tools/cli.py` should import domain operations from `mcp_servers` directly (preserving existing location) but expose all necessary CLI commands under a unified interface.

## 1. Command Inventory

### Chronicle Commands (`chronicle`)

| Action | Args | `domain_cli.py` | `tools/cli.py` | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **list** | `--limit` | ✅ | ✅ | **Parity** | Implemented. |
| **search** | `query` | ✅ | ✅ | **Parity** | Implemented. |
| **get** | `number` | ✅ | ✅ | **Parity** | Implemented. |
| **create** | `title`, `--content`, `--author`, `--status`, `--classification` | ✅ | ✅ | **Parity** | Implemented. |
| **update** | `number`, `--title`, `--content`, `--status`, `--reason` | ❌ | ✅ | **Extended** | NEW in tools/cli.py |

### Task Commands (`task`)

| Action | Args | `domain_cli.py` | `tools/cli.py` | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **list** | `--status` | ✅ | ✅ | **Parity** | Implemented. |
| **get** | `number` | ✅ | ✅ | **Parity** | Implemented. |
| **create** | `title`, `--objective`, `--deliverables`, `--acceptance-criteria`, `--priority`, `--status`, `--lead` | ✅ | ✅ | **Parity** | Implemented. |
| **update-status** | `number`, `new_status`, `--notes` | ✅ | ✅ | **Parity** | Implemented. |
| **search** | `query` | ❌ | ✅ | **Extended** | NEW in tools/cli.py |
| **update** | `number`, `--title`, `--objective`, `--priority`, `--lead` | ❌ | ✅ | **Extended** | NEW in tools/cli.py |

### ADR Commands (`adr`)

| Action | Args | `domain_cli.py` | `tools/cli.py` | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **list** | `--status` | ✅ | ✅ | **Parity** | Implemented. |
| **search** | `query` | ✅ | ✅ | **Parity** | Implemented. |
| **get** | `number` | ✅ | ✅ | **Parity** | Implemented. |
| **create** | `title`, `--context`, `--decision`, `--consequences`, `--status` | ✅ | ✅ | **Parity** | Implemented. |
| **update-status** | `number`, `new_status`, `--reason` | ❌ | ✅ | **Extended** | NEW in tools/cli.py |

### Protocol Commands (`protocol`)

| Action | Args | `domain_cli.py` | `tools/cli.py` | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **list** | `--status` | ✅ | ✅ | **Parity** | Implemented. |
| **search** | `query` | ✅ | ✅ | **Parity** | Implemented. |
| **get** | `number` | ✅ | ✅ | **Parity** | Implemented. |
| **create** | `title`, `--content`, `--version`, `--status`, `--authority`, `--classification` | ✅ | ✅ | **Parity** | Implemented. |
| **update** | `number`, `--title`, `--content`, `--status`, `--version`, `--reason` | ❌ | ✅ | **Extended** | NEW in tools/cli.py |

### Forge Commands (`forge`) [NEW]

| Action | Args | `domain_cli.py` | `tools/cli.py` | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **query** | `prompt`, `--temperature`, `--max-tokens`, `--system` | ❌ | ✅ | **New** | Requires ollama |
| **status** | — | ❌ | ✅ | **New** | Model availability |

## 2. Summary Statistics

| Category | In `domain_cli.py` | In `tools/cli.py` | Parity | Extended |
| :--- | :--- | :--- | :--- | :--- |
| Chronicle | 4 | 5 | ✅ 4 | +1 (update) |
| Task | 4 | 6 | ✅ 4 | +2 (search, update) |
| ADR | 4 | 5 | ✅ 4 | +1 (update-status) |
| Protocol | 4 | 5 | ✅ 4 | +1 (update) |
| Forge | 0 | 2 | ✅ N/A | +2 (query, status) |
| **Total** | **16** | **23** | **16 / 16** | **+7 extended** ✅ |

## 3. Shared Dependencies

| Dependency | Source | Notes |
| :--- | :--- | :--- |
| `ChronicleOperations` | `mcp_servers.chronicle.operations` | Full CRUD operations for chronicle entries. |
| `TaskOperations` | `mcp_servers.task.operations` | Full CRUD operations for tasks. |
| `ADROperations` | `mcp_servers.adr.operations` | Full CRUD operations for ADRs. |
| `ProtocolOperations` | `mcp_servers.protocol.operations` | Full CRUD operations for protocols. |
| `taskstatus`, `TaskPriority` | `mcp_servers.task.models` | Enum types for task status and priority. |
| `find_project_root` | `mcp_servers.lib.path_utils` | Shared path resolution. |
| `get_env_variable` | `mcp_servers.lib.env_helper` | Environment variable handling. |

## 4. Action Plan

### A. Commands to Add to `tools/cli.py`

1.  **`chronicle`** cluster with: `list`, `search`, `get`, `create`
2.  **`task`** cluster with: `list`, `get`, `create`, `update-status`
3.  **`adr`** cluster with: `list`, `search`, `get`, `create`
4.  **`protocol`** cluster with: `list`, `search`, `get`, `create`

### B. Imports to Add

```python
from mcp_servers.chronicle.operations import ChronicleOperations
from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import taskstatus, TaskPriority
from mcp_servers.adr.operations import ADROperations
from mcp_servers.protocol.operations import ProtocolOperations
```

### C. Deprecation

-   Once `tools/cli.py` has full parity, `scripts/domain_cli.py` can be deprecated and deleted.

## 5. Workflow Command Mapping

The `/workflow-*` commands in `.agent/workflows/` reference `domain_cli.py`. These will need to be updated to use `tools/cli.py` after migration:

| Workflow | Line | Current Command | Target Command |
| :--- | :--- | :--- | :--- |
| `adr-manage.md` | 8 | `python3 scripts/domain_cli.py adr list --limit 5` | `python3 tools/cli.py adr list --limit 5` |
| `adr-manage.md` | 11 | `python3 scripts/domain_cli.py adr create ...` | `python3 tools/cli.py adr create ...` |
| `sanctuary-chronicle.md` | 8 | `python3 scripts/domain_cli.py chronicle list --limit 5` | `python3 tools/cli.py chronicle list --limit 5` |
| `sanctuary-chronicle.md` | 11-12 | `python3 scripts/domain_cli.py chronicle create/update` | `python3 tools/cli.py chronicle create/update` |
| `tasks-manage.md` | 8 | `python3 scripts/domain_cli.py task list --status active` | `python3 tools/cli.py task list --status active` |
| `tasks-manage.md` | 11-12 | `python3 scripts/domain_cli.py task create/update` | `python3 tools/cli.py task create/update` |

> **Note**: `/adr-manage` uses `next_number.py` and templates, not `domain_cli.py` directly. Can be revisited later.

---

## 6. Full MCP Server Audit

Complete audit of `mcp_servers/` operations to identify any unmapped capabilities.

### Fully Mapped (via `tools/cli.py`)

| MCP Server | Operations Class | CLI Command | Status |
| :--- | :--- | :--- | :--- |
| `learning` | `LearningOperations` | `debrief`, `snapshot`, `persist-soul`, `guardian`, `rlm-distill` | ✅ Complete |
| `rag_cortex` | `CortexOperations` | `ingest`, `query`, `stats`, `cache-stats`, `cache-warmup` | ✅ Complete |
| `evolution` | `EvolutionOperations` | `evolution fitness/depth/scope` | ✅ Complete |
| `chronicle` | `ChronicleOperations` | `chronicle list/search/get/create` | ✅ Complete (NEW) |
| `task` | `TaskOperations` | `task list/get/create/update-status` | ✅ Complete (NEW) |
| `adr` | `ADROperations` | `adr list/search/get/create` | ✅ Complete (NEW) |
| `protocol` | `ProtocolOperations` | `protocol list/search/get/create` | ✅ Complete (NEW) |

### Partially Mapped or Unmapped

| MCP Server | Operations Class | Available Methods | CLI Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `git` | `GitOperations` | `add`, `commit`, `push`, `pull`, `status`, `diff`, `log`, `create_branch`, `checkout`, `start_feature`, `finish_feature`, `delete_local_branch`, `delete_remote_branch` | ⚠️ Partial | Used internally by `workflow end`. No direct CLI exposure. |
| `code` | `CodeOperations` | `lint`, `format_code`, `analyze`, `find_file`, `list_files`, `search_content`, `read_file`, `write_file`, `delete_file` | ❌ Unmapped | Potential future `code lint`, `code format` commands. |
| `orchestrator` | `OrchestratorOperations` | `create_cognitive_task`, `query_mnemonic_cortex`, `run_strategic_cycle` | ❌ Unmapped | Advanced orchestration - may not need CLI. |
| `council` | `CouncilOperations` | `dispatch_task`, `list_agents` | ❌ Unmapped | Council deliberation - used by Orchestrator. |
| `forge_llm` | `ForgeOperations` | `query_sanctuary_model`, `check_model_availability` | ❌ Unmapped | Model interaction - potential `forge query` command. |

### Minor Unmapped Methods (Domain Operations)

| Operation | Method | Status | Priority |
| :--- | :--- | :--- | :--- |
| `ChronicleOperations` | `update_entry` | ❌ Unmapped | Low - rarely used |
| `TaskOperations` | `update_task`, `search_tasks` | ⚠️ Partial | `update_task` generic updates not exposed, `search_tasks` unmapped |
| `ADROperations` | `update_adr_status` | ❌ Unmapped | Low - status changes rare |
| `ProtocolOperations` | `update_protocol` | ❌ Unmapped | Low - rarely used |

### Recommendation

The current implementation covers **all primary CRUD operations** for domain entities. The unmapped methods are either:
1. **Internal helpers** (used by other methods)
2. **Advanced orchestration** (Council, Forge) - not suitable for simple CLI
3. **Update variants** - can be added later if needed

**No critical gaps identified.**

