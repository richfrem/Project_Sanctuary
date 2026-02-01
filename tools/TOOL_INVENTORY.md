# Tool Inventory

> **Auto-generated:** 2026-02-01 10:07
> **Source:** [`tools/tool_inventory.json`](tools/tool_inventory.json)
> **Regenerate:** `python tools/curate/inventories/manage_tool_inventory.py generate --inventory tools/tool_inventory.json`

---

## 📁 Orchestrator

| Script | Description |
| :--- | :--- |
| [`workflow_manager.py`](tools/orchestrator/workflow_manager.py) | Manages the lifecycle of Agent Workflows (Start, Step, End). Tracks state in workflow_state.json. |

## 📦 Bundler

| Script | Description |
| :--- | :--- |
| [`bundle.py`](tools/retrieve/bundler/bundle.py) | Bundles multiple source files into a single Markdown 'Context Bundle' based on a JSON manifest. Warns on deprecated legacy keys. |
| [`manifest_manager.py`](tools/retrieve/bundler/manifest_manager.py) | Handles initialization and modification of the context-manager manifest. Acts as the primary CLI for the Context Bundler. Supports strict type validation. |
| [`validate.py`](tools/retrieve/bundler/validate.py) | Validates context bundler manifest files against schema. Checks required fields, path format, path traversal attacks, and legacy format warnings. |

## 📁 Cli Entry Points

| Script | Description |
| :--- | :--- |
| [`cli.py`](tools/cli.py) | Universal Tool & Workflow Router. The primary interface for Tool Discovery and Workflow Execution. |

## 📁 Curate

| Script | Description |
| :--- | :--- |
| [`manage_tool_inventory.py`](tools/curate/inventories/manage_tool_inventory.py) | Comprehensive manager for Tool Inventories. Supports list, add, update, remove, search, audit, and generate operations. |
| [`vibe_cleanup.py`](tools/curate/inventories/vibe_cleanup.py) | Maintenance script to clean up temporary/vibe files (logs, temp artifacts) from the workspace to ensure hygiene. |

## 📝 Documentation

| Script | Description |
| :--- | :--- |
| [`analyze_tracking_status.py`](tools/codify/tracking/analyze_tracking_status.py) | Generates a summary report of AI Analysis progress from the tracking file. Shows analyzed vs pending forms for project management dashboards. |
| [`capture-code-snapshot.js`](tools/codify/utils/capture-code-snapshot.js) | Generates a single text file snapshot of code files for LLM context sharing. |
| [`export_mmd_to_image.py`](tools/codify/diagrams/export_mmd_to_image.py) | Renders all .mmd files in docs/architecture_diagrams/ to PNG images. Run this script whenever diagrams are updated to regenerate images. |
| [`generate_todo_list.py`](tools/codify/tracking/generate_todo_list.py) | Creates a prioritized TODO list of forms pending AI analysis. Bubbles up Critical and High priority items based on workflow usage. |
| [`workflow_inventory_manager.py`](tools/curate/documentation/workflow_inventory_manager.py) | Manages the workflow inventory for agent workflows (.agent/workflows/*.md). Provides search, scan, add, and update capabilities. Outputs are docs/antigravity/workflow/workflow_inventory.json and docs/antigravity/workflow/WORKFLOW_INVENTORY.md. |

## 📁 Investigate

| Script | Description |
| :--- | :--- |
| [`next_number.py`](tools/investigate/utils/next_number.py) | Generates the next sequential ID number for project artifacts (ADRs, Tasks, Specs) by scanning the filesystem for existing files. |
| [`path_resolver.py`](tools/investigate/utils/path_resolver.py) | Standardizes cross-platform path resolution (Legacy Location). |

## 📁 Orchestrator

| Script | Description |
| :--- | :--- |
| [`proof_check.py`](tools/orchestrator/proof_check.py) | Validates task completion by checking file modifications against the Git index. Ensures Proof of Work. |

## 📁 Retrieve

| Script | Description |
| :--- | :--- |
| [`fetch_tool_context.py`](tools/retrieve/rlm/fetch_tool_context.py) | Retrieves the 'Gold Standard' tool definition from the RLM Tool Cache and formats it into an Agent-readable 'Manual Page'. This is the second step of the Late-Binding Protocol, following query_cache.py which finds a tool, this script provides the detailed context needed to use it. |

## 🧠 Rlm

| Script | Description |
| :--- | :--- |
| [`cleanup_cache.py`](tools/curate/rlm/cleanup_cache.py) | Prunes stale or orphaned entries from the RLM Cache to ensure it matches the filesystem state. |
| [`debug_rlm.py`](tools/codify/rlm/debug_rlm.py) | Debug utility to inspect the RLMConfiguration state. Verifies path resolution, manifest loading, and environment variable overrides. Useful for troubleshooting cache path conflicts. |
| [`distiller.py`](tools/codify/rlm/distiller.py) | Recursive summarization of repo content using Ollama. |
| [`inventory.py`](tools/retrieve/rlm/inventory.py) | RLM Auditor: Reports coverage of the semantic ledger against the filesystem. Uses the Shared RLMConfig to dynamically switch between 'Legacy' (Documentation) and 'Tool' (CLI) audit modes. |
| [`query_cache.py`](tools/retrieve/rlm/query_cache.py) | RLM Search: Instant O(1) semantic search of the ledger. |
| [`rlm_config.py`](tools/codify/rlm/rlm_config.py) | Central configuration factory for RLM. Resolves cache paths and loads manifests. |

## 🚀 Root

| Script | Description |
| :--- | :--- |
| [`cli.py`](tools/cli.py) | Universal Tool & Workflow Router. The primary interface for Tool Discovery and Workflow Execution. |
| [`cortex_cli.py`](scripts/cortex_cli.py) | Main CLI entry point for the Cortex Agentic Operations (Protocol 128). Orchestrates Cognitive functions (Memory, Learning, Debrief, Stats). |
| [`domain_cli.py`](scripts/domain_cli.py) | Domain-Specific CLI for managing specific Project Entities (Tasks, ADRs, Chronicles, Protocols). Maps CLI commands to MCP business logic. |
| [`smart_fix_links.py`](scripts/link-checker/smart_fix_links.py) | Auto-repair utility for broken Markdown links using a file inventory. |
| [`verify_links.py`](scripts/link-checker/verify_links.py) | Verifies the integrity of internal links across the documentation base. Part of Protocol 128 validation. |

## 🛠️ Utils

| Script | Description |
| :--- | :--- |
| [`path_resolver.py`](tools/utils/path_resolver.py) | Standardizes cross-platform path resolution and provides access to the Master Object Collection (MOC). Acts as a central utility for file finding. |
