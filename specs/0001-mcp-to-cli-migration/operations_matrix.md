# MCP Operations Migration Matrix

**Spec**: 0001-mcp-to-cli-migration  
**Created**: 2026-01-31  
**Last Updated**: 2026-01-31 21:35  
**Purpose**: Track migration of MCP operations to CLI with onboarding status

---

## Quick Stats

| Category | Count | Status |
|:---------|:------|:-------|
| Tools Registered in Inventory | 25 | ✅ Active |
| Tools in RLM Cache | 25 | ✅ Synced |
| Workflows Converted to Python CLI | 24/24 | ✅ Done |
| Out of Scope (Skipped) | ~43 ops | ❌ Deferred |

---

## IN SCOPE: CLI & Tool Onboarding Tracker

### Cluster: Core CLIs (Root Level)

| Tool | Header | Inventory | RLM Cache | Workflow | Notes |
|:-----|:------:|:---------:|:---------:|:--------:|:------|
| `scripts/domain_cli.py` | ✅ | ✅ | ✅ | N/A | Chronicle, Task, ADR, Protocol |
| `scripts/cortex_cli.py` | ✅ | ✅ | ✅ | N/A | P128, Evolution, RAG |
| `tools/cli.py` | ✅ | ✅ | ✅ | N/A | Workflow orchestration |

### Cluster: Retrieve (Tool Discovery)

| Tool | Header | Inventory | RLM Cache | Workflow | Notes |
|:-----|:------:|:---------:|:---------:|:--------:|:------|
| `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py` | ✅ | ✅ | ✅ | N/A | Tool discovery (Late-Binding) |
| `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py` | ✅ | ✅ | ✅ | N/A | Context fetch after discovery |
| `plugins/context-bundler/scripts/bundle.py` | ✅ | ✅ | ✅ | N/A | Context bundling |
| `plugins/context-bundler/scripts/bundle.py` | ✅ | ✅ | ✅ | N/A | Manifest management |

### Cluster: Codify (Content Creation)

| Tool | Header | Inventory | RLM Cache | Workflow | Notes |
|:-----|:------:|:---------:|:---------:|:--------:|:------|
| `plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py` | ✅ | ✅ | ✅ | N/A | RLM summary distillation |
| `plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py` | ✅ | ✅ | ✅ | N/A | RLM configuration |
| `plugins/rlm-factory/skills/rlm-curator/scripts/debug_rlm.py` | ✅ | ✅ | ✅ | N/A | RLM debugging |
| `plugins/mermaid-to-png/skills/convert-mermaid/scripts/convert.py` | ✅ | ✅ | ✅ | N/A | Mermaid to PNG |
| `plugins/tool-inventory/skills/tool-inventory/scripts/audit_plugins.py` | ✅ | ✅ | ✅ | N/A | Task tracking analysis |
| `plugins/task-manager/skills/task-agent/scripts/create_task.py` | ✅ | ✅ | ✅ | N/A | TODO generation |

### Cluster: Curate (Maintenance)

| Tool | Header | Inventory | RLM Cache | Workflow | Notes |
|:-----|:------:|:---------:|:---------:|:--------:|:------|
| `plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py` | ✅ | ✅ | ✅ | N/A | Tool registration |
| `plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py` | ✅ | ✅ | ✅ | N/A | Session cleanup |
| `plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py` | ✅ | ✅ | ✅ | N/A | RLM cache maintenance |
| `plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py` | ✅ | ✅ | ✅ | N/A | Workflow docs |

### Cluster: Investigate (Analysis)

| Tool | Header | Inventory | RLM Cache | Workflow | Notes |
|:-----|:------:|:---------:|:---------:|:--------:|:------|
| `plugins/adr-manager/skills/adr-management/scripts/next_number.py` | ✅ | ✅ | ✅ | N/A | Sequential numbering for ADRs/Specs |

### Cluster: Orchestrator (Workflow Execution)

| Tool | Header | Inventory | RLM Cache | Workflow | Notes |
|:-----|:------:|:---------:|:---------:|:--------:|:------|
| `plugins/agent-loops/skills/orchestrator/scripts/proof_check.py` | ✅ | ✅ | ✅ | N/A | Task proof verification |
| `tools/orchestrator/workflow_manager.py` | ✅ | ✅ | ✅ | N/A | Workflow state management |

---

## IN SCOPE: MCP Operations → CLI Mapping

### Cluster: sanctuary_cortex (Protocol 128) - HIGH VALUE

| MCP Operation | CLI Command | Status | Notes |
|:--------------|:------------|:------:|:------|
| `cortex-query` | `cortex_cli.py query` | ✅ Done | RAG query |
| `cortex-ingest-full` | `cortex_cli.py ingest` | ✅ Done | Full reindex |
| `cortex-ingest-incremental` | `cortex_cli.py ingest --incremental` | ✅ Done | Incremental |
| `cortex-get-stats` | `cortex_cli.py stats` | ✅ Done | DB health |
| `cortex-guardian-wakeup` | `cortex_cli.py guardian` | ✅ Done | P128 |
| `cortex-learning-debrief` | `cortex_cli.py debrief` | ✅ Done | P128 |
| `cortex-capture-snapshot` | `cortex_cli.py snapshot` | ✅ Done | P128 |
| `cortex-persist-soul` | `cortex_cli.py persist-soul` | ✅ Done | ADR 079 |
| `cortex-cache-*` (4 tools) | — | 🔬 Analyze | CAG vs RLM cache overlap |

### Cluster: sanctuary_domain - HIGH VALUE

| MCP Operation | CLI Command | Status | Notes |
|:--------------|:------------|:------:|:------|
| `chronicle-list-entries` | `domain_cli.py chronicle list` | ✅ Done | |
| `chronicle-search` | `domain_cli.py chronicle search` | ✅ Done | |
| `chronicle-get-entry` | `domain_cli.py chronicle get` | ✅ Done | |
| `chronicle-create-entry` | `domain_cli.py chronicle create` | ✅ Done | |
| `protocol-list` | `domain_cli.py protocol list` | ✅ Done | |
| `protocol-search` | `domain_cli.py protocol search` | ✅ Done | |
| `protocol-get` | `domain_cli.py protocol get` | ✅ Done | |
| `protocol-create` | `domain_cli.py protocol create` | ✅ Done | |
| `task-list` | `domain_cli.py task list` | ✅ Done | |
| `task-update-status` | `domain_cli.py task update-status` | ✅ Done | |
| `task-get` | `domain_cli.py task get` | ✅ Done | |
| `task-create` | `domain_cli.py task create` | ✅ Done | |
| `adr-list` | `domain_cli.py adr list` | ✅ Done | |
| `adr-search` | `domain_cli.py adr search` | ✅ Done | |
| `adr-get` | `domain_cli.py adr get` | ✅ Done | |
| `adr-create` | `domain_cli.py adr create` | ✅ Done | |


---

## NEEDS DEEPER ANALYSIS

| Item | Reason | Priority |
|:-----|:-------|:---------|
| RLM vs CAG Cache | ✅ ADR-097: Segregation of Duties | High |
| Forge LLM (`query-sanctuary-model`) | ⏳ Backlog: Low Priority | Low |

---

## OUT OF SCOPE (Deferred)

These operations are intentionally excluded from migration because native equivalents exist or functionality is deprecated.

### Cluster: sanctuary_utils (17 tools) - Native Python

| Operation | Reason for Skip |
|:----------|:----------------|
| `time-get-current-time` | Python `datetime` / `date` command |
| `time-get-timezone-info` | Python `datetime` / system commands |
| `calculator-*` (5 tools) | Python can do math natively |
| `uuid-*` (3 tools) | Python `uuid` module |
| `string-*` (6 tools) | Python string methods |
| `gateway-get-capabilities` | Gateway-specific metadata |

### Cluster: sanctuary_filesystem (11 tools) - Antigravity Native

| Operation | Reason for Skip |
|:----------|:----------------|
| `code-read` | Antigravity `view_file` |
| `code-write` | Antigravity `write_to_file` |
| `code-delete` | Antigravity tools (risky) |
| `code-get-info` | Antigravity `list_dir` |
| `code-list-files` | Antigravity `find_by_name` |
| `code-find-file` | Antigravity `find_by_name` |
| `code-search-content` | Antigravity `grep_search` |
| `code-lint`, `format`, `analyze` | Native linters (low priority) |

### Cluster: sanctuary_network (2 tools) - Antigravity Native

| Operation | Reason for Skip |
|:----------|:----------------|
| `fetch-url` | Antigravity `read_url_content` |
| `check-site-status` | `curl` command |

### Legacy MCP Servers (Deprecated)

| Server | Operations | Reason for Skip |
|:-------|:-----------|:----------------|
| **Council MCP** | 2 | Multi-agent deliberation - not in current use |
| **Orchestrator MCP** | 2 | Strategic missions - may revisit later |
| **Agent Persona** | 5 | Multi-agent roles - not in current use |
| **Config MCP** | 4 | Direct file access suffices |

### Cluster: sanctuary_git - MAYBE

| MCP Operation | CLI Command | Status | Notes |
|:--------------|:------------|:------:|:------|
| `git-get-status` | — | ❓ Eval | Native `git status` works |
| `git-smart-commit` | — | ❓ Eval | Protocol 101 value |
| `git-start-feature` | — | ❓ Eval | Workflow automation |
| `git-push-feature` | — | ❓ Eval | Workflow automation |
| `git-finish-feature` | — | ❓ Eval | Workflow automation |

---

## Legend

| Symbol | Meaning |
|:-------|:--------|
| ✅ | Complete |
| ⚠️ Needs | Needs update (header/docs) |
| ⏳ TODO | Not yet implemented |
| 🔬 Analyze | Needs deeper investigation |
| ❓ Eval | Evaluate if needed |
| ❌ | Out of scope / Skipped |
