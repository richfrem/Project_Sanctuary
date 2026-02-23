# Tool Inventory

> **Auto-generated:** 2026-02-22 16:45
> **Source:** [`tools/tool_inventory.json`](tools/tool_inventory.json)
> **Regenerate:** `python plugins/tool-inventory/scripts/manage_tool_inventory.py generate --inventory tools/tool_inventory.json`

---

## 📁 Adr-Manager

| Script | Description |
| :--- | :--- |
| [`create_adr.py`](plugins/adr-manager/skills/adr-management/scripts/create_adr.py) | create_adr.py (CLI) |
| [`next_number.py`](plugins/adr-manager/skills/adr-management/scripts/next_number.py) | next_number.py (CLI) |

## 📁 Agent-Loops

| Script | Description |
| :--- | :--- |
| [`agent_orchestrator.py`](plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py) | Agent Orchestrator (Core Script) |
| [`closure-guard.sh`](plugins/agent-loops/hooks/closure-guard.sh) | TBD |
| [`proof_check.py`](plugins/agent-loops/skills/orchestrator/scripts/proof_check.py) | Proof Check (CLI) |

## 📁 Agent-Scaffolders

| Script | Description |
| :--- | :--- |
| [`audit.py`](plugins/agent-scaffolders/scripts/audit.py) | audit.py (CLI) |
| [`scaffold.py`](plugins/agent-scaffolders/scripts/scaffold.py) | scaffold.py (CLI) |

## 📁 Cli Entry Points

| Script | Description |
| :--- | :--- |
| [`cli.py`](tools/cli.py) | Universal Tool & Workflow Router. The primary interface for Tool Discovery and Workflow Execution. |

## 📁 Coding-Conventions

| Script | Description |
| :--- | :--- |
| [`js-tool-header-template.js`](plugins/coding-conventions/templates/js-tool-header-template.js) |  |
| [`python-tool-header-template.py`](plugins/coding-conventions/templates/python-tool-header-template.py) | {{script_name}} (CLI) |

## 📁 Context-Bundler

| Script | Description |
| :--- | :--- |
| [`bundle.py`](plugins/context-bundler/scripts/bundle.py) | Context Bundler Engine |
| [`manifest_manager.py`](plugins/context-bundler/scripts/manifest_manager.py) | TBD |
| [`path_resolver.py`](plugins/context-bundler/scripts/path_resolver.py) | path_resolver.py (CLI) |

## 📁 Excel-To-Csv

| Script | Description |
| :--- | :--- |
| [`convert.py`](plugins/excel-to-csv/skills/excel-to-csv/scripts/convert.py) | excel_to_csv.py (CLI) |

## 🔗 Link-Checker

| Script | Description |
| :--- | :--- |
| [`check_broken_paths.py`](plugins/link-checker/skills/link-checker-agent/scripts/check_broken_paths.py) | check_broken_paths.py (CLI) |
| [`enrich_links_v2.py`](plugins/link-checker/skills/link-checker-agent/scripts/enrich_links_v2.py) | enrich_links_v2.py (CLI) |
| [`find_source_links.py`](plugins/link-checker/skills/link-checker-agent/scripts/find_source_links.py) | find_source_links.py (CLI) |
| [`map_repository_files.py`](plugins/link-checker/skills/link-checker-agent/scripts/map_repository_files.py) | map_repository_files.py (CLI) |
| [`smart_fix_links.py`](plugins/link-checker/skills/link-checker-agent/scripts/smart_fix_links.py) | smart_fix_links.py (CLI) |

## 📁 Mermaid-To-Png

| Script | Description |
| :--- | :--- |
| [`convert.py`](plugins/mermaid-to-png/skills/convert-mermaid/scripts/convert.py) | convert.py (CLI) |

## 📁 Plugin-Manager

| Script | Description |
| :--- | :--- |
| [`audit_structure.py`](plugins/plugin-manager/scripts/audit_structure.py) | Audit Plugin Structure |
| [`bulk_replicator.py`](plugins/plugin-manager/scripts/bulk_replicator.py) | Update All Plugins |
| [`clean_orphans.py`](plugins/plugin-manager/scripts/clean_orphans.py) | Plugin Orphan Cleaner |
| [`cleanup_targets.py`](plugins/plugin-manager/scripts/cleanup_targets.py) | TBD |
| [`generate_readmes.py`](plugins/plugin-manager/scripts/generate_readmes.py) | Generate Plugin READMEs |
| [`plugin_bootstrap.py`](plugins/plugin-manager/scripts/plugin_bootstrap.py) | plugin_bootstrap.py (CLI) |
| [`plugin_inventory.py`](plugins/plugin-manager/scripts/plugin_inventory.py) | Plugin Inventory Generator |
| [`plugin_replicator.py`](plugins/plugin-manager/scripts/plugin_replicator.py) | Plugin Replicator |
| [`sync_with_inventory.py`](plugins/plugin-manager/scripts/sync_with_inventory.py) | Sync Plugins with Inventory |
| [`update_agent_system.py`](plugins/plugin-manager/scripts/update_agent_system.py) | Update Agent System (Master Sync) |

## 📁 Plugin-Mapper

| Script | Description |
| :--- | :--- |
| [`bridge_installer.py`](plugins/plugin-mapper/skills/agent-bridge/scripts/bridge_installer.py) | bridge_installer.py (CLI) |
| [`install_all_plugins.py`](plugins/plugin-mapper/skills/agent-bridge/scripts/install_all_plugins.py) | install_all_plugins.py (CLI) |

## 📁 Rlm-Factory

| Script | Description |
| :--- | :--- |
| [`cleanup_cache.py`](plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py) | cleanup_cache.py |
| [`debug_rlm.py`](plugins/rlm-factory/skills/rlm-curator/scripts/debug_rlm.py) | debug_rlm.py |
| [`distiller.py`](plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py) | distiller.py |
| [`inventory.py`](plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py) | inventory.py |
| [`query_cache.py`](plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py) | query_cache.py |
| [`rlm_config.py`](plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py) | rlm_config.py |

## 📁 Spec-Kitty-Plugin

| Script | Description |
| :--- | :--- |
| [`sync_configuration.py`](plugins/spec-kitty-plugin/skills/spec-kitty-agent/scripts/sync_configuration.py) | Spec Kitty Configuration Sync |
| [`verify_workflow_state.py`](plugins/spec-kitty-plugin/skills/spec-kitty-agent/scripts/verify_workflow_state.py) | verify_workflow_state.py |

## 📁 Task-Manager

| Script | Description |
| :--- | :--- |
| [`board.py`](plugins/task-manager/skills/task-agent/scripts/board.py) | board.py (CLI) |
| [`create_task.py`](plugins/task-manager/skills/task-agent/scripts/create_task.py) | create_task.py (CLI) |

## 📁 Tool-Inventory

| Script | Description |
| :--- | :--- |
| [`audit_plugins.py`](plugins/tool-inventory/skills/tool-inventory/scripts/audit_plugins.py) | Audit Plugin Inventory |
| [`fix_inventory_paths.py`](plugins/tool-inventory/skills/tool-inventory/scripts/fix_inventory_paths.py) | Fix Inventory Paths |
| [`generate_tools_manifest.py`](plugins/tool-inventory/skills/tool-inventory/scripts/generate_tools_manifest.py) | Purpose: |
| [`manage_tool_inventory.py`](plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py) | TBD |
| [`rebuild_inventory.py`](plugins/tool-inventory/skills/tool-inventory/scripts/rebuild_inventory.py) | rebuild_inventory.py |
| [`sync_inventory_descriptions.py`](plugins/tool-inventory/skills/tool-inventory/scripts/sync_inventory_descriptions.py) | sync_inventory_descriptions.py |
| [`tool_chroma.py`](plugins/tool-inventory/skills/tool-inventory/scripts/tool_chroma.py) | tool_chroma.py — Embedded ChromaDB wrapper for tool-inventory plugin |
| [`tool_inventory_init.py`](plugins/tool-inventory/skills/tool-inventory-init/scripts/tool_inventory_init.py) | Tool Inventory Setup (Librarian Bootstrapper) |

## 📁 Vector-Db

| Script | Description |
| :--- | :--- |
| [`cleanup.py`](plugins/vector-db/skills/vector-db-agent/scripts/cleanup.py) | cleanup.py (CLI) |
| [`ingest.py`](plugins/vector-db/skills/vector-db-agent/scripts/ingest.py) | ingest.py (CLI) |
| [`ingest_code_shim.py`](plugins/vector-db/skills/vector-db-agent/scripts/ingest_code_shim.py) | ingest_code_shim.py (CLI) |
| [`init.py`](plugins/vector-db/skills/vector-db-init/scripts/init.py) | init.py (CLI) |
| [`operations.py`](plugins/vector-db/skills/vector-db-agent/scripts/operations.py) | Core domain logic for Vector DB operations. |
| [`query.py`](plugins/vector-db/skills/vector-db-agent/scripts/query.py) | query.py (CLI) |
| [`vector_config.py`](plugins/vector-db/skills/vector-db-agent/scripts/vector_config.py) | Wraps the raw manifest JSON dict and provides file discovery methods. |
| [`vector_consistency_check.py`](plugins/vector-db/skills/vector-db-agent/scripts/vector_consistency_check.py) | Vector Consistency Stabilizer |
