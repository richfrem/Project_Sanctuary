# Workflow Inventory & Analysis

**Objective**: Audit and classify all Antigravity workflows into the **Dual-Track** architecture (Factory vs. Discovery).

**Created**: 2026-01-30  
**Updated**: 2026-01-31  
**Status**: ACTIVE (Spec 0001 - MCP-to-CLI Migration)

## The Dual-Track Taxonomy

*   **Track A (Factory)**: Deterministic, repeatable, high-volume. The Agent follows a strict script. "SOPs".
*   **Track B (Discovery)**: High-ambiguity, creative, exploratory. The Agent uses `spec.md` and `plan.md` to define its own path. "Spec-Driven".
*   **Shared (Meta-Ops)**: Operational workflows that support both tracks (e.g., git ops, retrospectives).

## Current Workflow Inventory (24 Workflows)

| Workflow Name | Classification | Pre-Flight Status | MCP/Bash Dependencies | Migration Status |
| :--- | :--- | :--- | :--- | :--- |
| **Core Meta-Ops (Tier 1)** | | | | |
| `/workflow-start` | Shared (Tier 1) | ✅ Python CLI | None | ✅ Complete |
| `/workflow-end` | Shared (Tier 1) | ✅ Python CLI | None | ✅ Complete |
| `/workflow-retrospective` | Shared (Meta-Ops) | ✅ Python CLI | Bash wrapper (thin) | ✅ Complete |
| **Documentation Factory (Track A)** | | | | |
| `/codify-adr` | Track A (Factory) | ✅ Python CLI | None | ✅ Complete |
| **Spec-Kit Core (Track B)** | | | | |
| `/speckit-specify` | Track B (Discovery) | ✅ Python CLI | None | ✅ Complete |
| `/speckit-clarify` | Track B (Discovery) | ✅ Python CLI | `check-prerequisites.sh` | ⚠️ Partial |
| `/speckit-plan` | Track B (Discovery) | ✅ Python CLI | `setup-plan.sh`, `update-agent-context.sh` | ⚠️ Partial |
| `/speckit-tasks` | Track B (Discovery) | ✅ Python CLI | `check-prerequisites.sh` | ⚠️ Partial |
| `/speckit-implement` | Track B (Discovery) | ✅ Python CLI | `check-prerequisites.sh` | ⚠️ Partial |
| `/speckit-constitution` | Track B (Discovery) | ✅ Python CLI | None | ✅ Complete |
| `/speckit-checklist` | Track B (Discovery) | ✅ Python CLI | `check-prerequisites.sh` | ⚠️ Partial |
| `/speckit-analyze` | Track B (Discovery) | ✅ Python CLI | `check-prerequisites.sh` | ⚠️ Partial |
| `/speckit-tasks-to-issues` | Track B (Discovery) | ✅ Python CLI | `check-prerequisites.sh`, **github-mcp-server** | ⚠️ Partial + MCP |
| **Protocol 128 (Learning)** | | | | |
| `/recursive_learning` | Shared (Meta-Ops) | ⚠️ Mixed | **Cortex MCP Suite** | 🔄 SOP Only |
| `/workflow-learning-loop` | Track A (SOP) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-scout` | Track A (SOP) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-audit` | Track A (SOP) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-seal` | Track A (SOP) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-persist` | Track A (SOP) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-ingest` | Track A (SOP) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-chronicle` | Shared (Meta-Ops) | ✅ Python CLI | Shim | ✅ Complete |
| `/workflow-task` | Shared (Meta-Ops) | ✅ Python CLI | Shim | ✅ Complete |
| **Utilities** | | | | |
| `/post-move-link-check` | Shared (Meta-Ops) | ⚠️ Manual | Bash commands | ⏳ Low Priority |

## Existing CLI Entry Points

The following CLIs already expose MCP operations and can be extended:

| CLI | Location | Domains | Status |
| :--- | :--- | :--- | :--- |
| `domain_cli.py` | `scripts/domain_cli.py` | Chronicle, Task, ADR, Protocol | ✅ Already imports `mcp_servers/` |
| `cortex_cli.py` | `scripts/cortex_cli.py` | RAG, Evolution, RLM, Learning | ✅ Already imports `mcp_servers/` |
| `cli.py` | `tools/cli.py` | Workflow orchestration | ⚠️ Needs extension |

## MCP Domains → CLI/Workflow Mapping (Proposed)

| MCP Server | Operations Count | CLI Entry Point | Workflow(s) Needed |
| :--- | :---: | :--- | :--- |
| Chronicle MCP | 7 | `domain_cli.py chronicle` ✅ | `/chronicle-*` (optional) |
| Task MCP | 6 | `domain_cli.py task` ✅ | `/task-*` (optional) |
| ADR MCP | 5 | `domain_cli.py adr` ✅ | `/codify-adr` (exists?) |
| Protocol MCP | 5 | `domain_cli.py protocol` ✅ | `/protocol-*` (optional) |
| RAG Cortex MCP | 8 | `cortex_cli.py` ✅ | `/retrieve-*` (exists?) |
| Evolution MCP | 5 | `cortex_cli.py evolution` ✅ | Part of learning loop |
| Git MCP | 8 | `git` (native) | `/workflow-end`, `/workflow-start` |
| Config MCP | 4 | TBD | `/config-*` (new) |
| Code MCP | 11 | TBD | `/investigate-*` (exists?) |
| Forge LLM MCP | 2 | TBD | Part of Discovery workflows |
| Agent Persona MCP | 5 | TBD | Part of council/orchestrator |
| Council MCP | 2 | TBD | `/council-*` (new, if needed) |
| Orchestrator MCP | 2 | `tools/cli.py workflow` | `/workflow-*` (exists) |
| Workflow MCP | 2 | `tools/cli.py workflow` | Merged into orchestrator |
| Learning MCP | ~5 | `cortex_cli.py` | `/recursive_learning` |

## Migration Strategy

1.  **Register all MCP operations in RLM tool cache** using `manage_tool_inventory.py`
2.  **Keep code in `mcp_servers/lib/`** - do not rewrite working operations
3.  **Extend existing CLIs** to cover any missing operations
4.  **Update workflows** to use Python CLI instead of `source scripts/bash/`
5.  **Create new workflows** only where needed (most domains have CLI already)

## Analysis Findings

1.  **Clear Separation**: The `codify-*` vs `speckit-*` namespace is resolved. `codify` is for **documenting what exists** (Factory). `speckit` is for **building what's new** (Discovery).
2.  **The "Bridge"**: The `investigate-*` workflows are crucial. They are "Factory" modules, but `speckit-plan` (Track B) should rely on them to gather context.
3.  **Existing CLIs**: `domain_cli.py` and `cortex_cli.py` already wrap most MCP operations - we just need to register them in the tool cache!
4.  **Thin Shims OK**: `workflow-start.sh`, `workflow-end.sh`, `workflow-retrospective.sh` are already thin pass-throughs. Keep them.

## Bash Script Migration Priority

| Script | Size | Priority | Notes |
| :--- | :--- | :--- | :--- |
| `update-agent-context.sh` | 26KB | **High** | Complex, needs Python port |
| `create-new-feature.sh` | 10KB | **High** | Merge into `WorkflowManager` |
| `check-prerequisites.sh` | 5KB | **Medium** | Used by many workflows |
| `common.sh` | 5KB | **Medium** | Shared utilities |
| `setup-plan.sh` | 2KB | **Low** | Part of `/speckit-plan` |
| `workflow-start.sh` | 655B | ✅ Keep | Thin shim |
| `workflow-end.sh` | 314B | ✅ Keep | Thin shim |
| `workflow-retrospective.sh` | 334B | ✅ Keep | Thin shim |

## Recommendations

1.  **Immediate**: Run `manage_tool_inventory.py discover` to find unregistered scripts
2.  **Phase 1**: Register `domain_cli.py` and `cortex_cli.py` commands in RLM cache
3.  **Phase 2**: Update speckit workflows to use `python tools/cli.py` pre-flight
4.  **Phase 3**: Port complex bash scripts (`update-agent-context.sh`, `create-new-feature.sh`)
5.  **Update `constitution.md`** with CLI-first rules
