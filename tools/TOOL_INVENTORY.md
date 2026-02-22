# Tool Inventory

> **Auto-generated:** 2026-02-01 14:52
> **Source:** [`plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json`](plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json)
> **Regenerate:** `python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py generate --inventory plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json`

---

## 📁 Orchestrator

| Script | Description |
| :--- | :--- |
| [`workflow_manager.py`](tools/orchestrator/workflow_manager.py) | Manages the lifecycle of Agent Workflows (Start, Step, End). Tracks state in workflow_state.json. |

## 📦 Bundler

| Script | Description |
| :--- | :--- |
| [`bundle.py`](plugins/context-bundler/scripts/bundle.py) | Bundles multiple source files into a single Markdown 'Context Bundle' based on a JSON manifest. Warns on deprecated legacy keys. |
| [`manifest_manager.py`](plugins/context-bundler/scripts/bundle.py) | Handles initialization and modification of the context-manager manifest. Acts as the primary CLI for the Context Bundler. Supports strict type validation. |
| [`validate.py`](plugins/context-bundler/scripts/bundle.py) | Validates context bundler manifest files against schema. Checks required fields, path format, path traversal attacks, and legacy format warnings. |

## 📁 Cli Entry Points

| Script | Description |
| :--- | :--- |
| [`cli.py`](tools/cli.py) | Main entry point for the Antigravity Command System. Supports Context Bundling, Tool Discovery, and Protocol 128 Learning Operations (Snapshot, Debrief, Guardian, Soul Persistence). Decoupled from mcp_servers. |

## 📁 Curate

| Script | Description |
| :--- | :--- |
| [`manage_tool_inventory.py`](plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py) | Comprehensive manager for Tool Inventories. Supports list, add, update, remove, search, audit, and generate operations. |
| [`vibe_cleanup.py`](plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py) | Maintenance script to clean up temporary/vibe files (logs, temp artifacts) from the workspace to ensure hygiene. |

## 📝 Documentation

| Script | Description |
| :--- | :--- |
| [`analyze_tracking_status.py`](plugins/tool-inventory/skills/tool-inventory/scripts/audit_plugins.py) | Generates a summary report of AI Analysis progress from the tracking file. Shows analyzed vs pending forms for project management dashboards. |
| [`capture_code_snapshot.py`](scripts/capture_code_snapshot.py) | Generates a single text file snapshot of code files for LLM context sharing. Direct Python port of the legacy Node.js utility. |
| [`export_mmd_to_image.py`](plugins/mermaid-to-png/skills/convert-mermaid/scripts/convert.py) | Renders all .mmd files in docs/architecture_diagrams/ to PNG images. Run this script whenever diagrams are updated to regenerate images. |
| [`generate_todo_list.py`](plugins/task-manager/skills/task-agent/scripts/create_task.py) | Creates a prioritized TODO list of forms pending AI analysis. Bubbles up Critical and High priority items based on workflow usage. |
| [`workflow_inventory_manager.py`](plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py) | Manages the workflow inventory for agent workflows (.agent/workflows/*.md). Provides search, scan, add, and update capabilities. Outputs are docs/antigravity/workflow/workflow_inventory.json and docs/antigravity/workflow/WORKFLOW_INVENTORY.md. |

## 📁 Investigate

| Script | Description |
| :--- | :--- |
| [`next_number.py`](plugins/adr-manager/skills/adr-management/scripts/next_number.py) | Generates the next sequential ID number for project artifacts (ADRs, Tasks, Specs) by scanning the filesystem for existing files. |
| [`path_resolver.py`](plugins/adr-manager/skills/adr-management/scripts/path_resolver.py) | Standardizes cross-platform path resolution (Legacy Location). |

## 📁 Orchestrator

| Script | Description |
| :--- | :--- |
| [`proof_check.py`](plugins/agent-loops/skills/orchestrator/scripts/proof_check.py) | Validates task completion by checking file modifications against the Git index. Ensures Proof of Work. |

## 📁 Retrieve

| Script | Description |
| :--- | :--- |
| [`fetch_tool_context.py`](plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py) | Retrieves the 'Gold Standard' tool definition from the RLM Tool Cache and formats it into an Agent-readable 'Manual Page'. This is the second step of the Late-Binding Protocol, following query_cache.py which finds a tool, this script provides the detailed context needed to use it. |

## 🧠 Rlm

| Script | Description |
| :--- | :--- |
| [`cleanup_cache.py`](plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py) | Prunes stale or orphaned entries from the RLM Cache to ensure it matches the filesystem state. |
| [`debug_rlm.py`](plugins/rlm-factory/skills/rlm-curator/scripts/debug_rlm.py) | Debug utility to inspect the RLMConfiguration state. Verifies path resolution, manifest loading, and environment variable overrides. Useful for troubleshooting cache path conflicts. |
| [`distiller.py`](plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py) | Recursive summarization of repo content using Ollama. |
| [`inventory.py`](plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py) | RLM Auditor: Reports coverage of the semantic ledger against the filesystem. Uses the Shared RLMConfig to dynamically switch between 'Legacy' (Documentation) and 'Tool' (CLI) audit modes. |
| [`query_cache.py`](plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py) | RLM Search: Instant O(1) semantic search of the ledger. |
| [`rlm_config.py`](plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py) | Central configuration factory for RLM. Resolves cache paths and loads manifests. |

## 🚀 Root

| Script | Description |
| :--- | :--- |
| [`cli.py`](tools/cli.py) | Universal Tool & Workflow Router. The primary interface for Tool Discovery and Workflow Execution. |
| [`cortex_cli.py`](scripts/cortex_cli.py) | Main CLI entry point for the Cortex Agentic Operations (Protocol 128). Orchestrates Cognitive functions (Memory, Learning, Debrief, Stats). |
| [`domain_cli.py`](scripts/domain_cli.py) | Domain-Specific CLI for managing specific Project Entities (Tasks, ADRs, Chronicles, Protocols). Maps CLI commands to MCP business logic. |
| [`env_helper.py`](mcp_servers/lib/env_helper.py) | Simple environment variable helper with proper fallback (Env -> .env). Ensures consistent secret loading across Project Sanctuary. |
| [`hf_decorate_readme.py`](scripts/hugging-face/hf_decorate_readme.py) | Prepares the local Hugging Face staging directory for upload. Modifies 'hugging_face_dataset_repo/README.md' in-place with YAML frontmatter per ADR 081. |
| [`hf_upload_assets.py`](scripts/hugging-face/hf_upload_assets.py) | Synchronizes staged landing-page assets with the Hugging Face Hub (ADR 081). Uploads the final, metadata-rich README.md to the repository root. |
| [`hf_utils.py`](mcp_servers/lib/hf_utils.py) | Hugging Face utility library for soul persistence (ADR 079). Encapsulates huggingface_hub logic. Provides unified async primitives for uploading files, folders, and updating datasets. |
| [`smart_fix_links.py`](scripts/link-checker/smart_fix_links.py) | Auto-repair utility for broken Markdown links using a file inventory. |
| [`upload_to_huggingface.py`](forge/scripts/upload_to_huggingface.py) | Manages the upload of model weights, GGUF files, and metadata to Hugging Face Hub (Phase 6). Handles artifact selection, repo creation, and secure transport. |
| [`verify_links.py`](scripts/link-checker/verify_links.py) | Verifies the integrity of internal links across the documentation base. Part of Protocol 128 validation. |

## 🛠️ Utils

| Script | Description |
| :--- | :--- |
| [`path_resolver.py`](plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py) | Standardizes cross-platform path resolution and provides access to the Master Object Collection (MOC). Acts as a central utility for file finding. |
