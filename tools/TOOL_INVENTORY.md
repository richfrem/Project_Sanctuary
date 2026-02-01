# Tool Inventory

> **Auto-generated:** 2026-01-31 19:30
> **Source:** [`tools/tool_inventory.json`](tools/tool_inventory.json)
> **Regenerate:** `python tools/curate/inventories/manage_tool_inventory.py generate --inventory tools/tool_inventory.json`

---

## 📁 Orchestrator

| Script | Description |
| :--- | :--- |
| [`workflow_manager.py`](tools/orchestrator/workflow_manager.py) | Core logic for the 'Python Orchestrator' architecture (ADR-0030 v2/v3). Handles Git State checks, Context Alignment, Branch Creation & Naming, and Context Manifest Initialization. Acts as the single source of truth for 'Start Workflow' logic. |

## 📦 Bundler

| Script | Description |
| :--- | :--- |
| [`bundle.py`](tools/retrieve/bundler/bundle.py) | Bundles multiple source files into a single Markdown 'Context Bundle' based on a JSON manifest. |
| [`manifest_manager.py`](tools/retrieve/bundler/manifest_manager.py) | Handles initialization and modification of the context-manager manifest. Acts as the primary CLI for the Context Bundler. |

## 📁 Cli Entry Points

| Script | Description |
| :--- | :--- |
| [`cli.py`](tools/cli.py) | Main entry point for the Antigravity Command System. Orchestrates sub-tools for scanning, context bundling, dependency analysis, and business rule discovery. Routes commands like scan, bundle, query, and rules to specialized modules. |

## 📁 Curate

| Script | Description |
| :--- | :--- |
| [`manage_tool_inventory.py`](tools/curate/inventories/manage_tool_inventory.py) | Comprehensive manager for Tool Inventories. Supports list, add, update, remove, search, audit, and generate operations. |

## 📝 Documentation

| Script | Description |
| :--- | :--- |
| [`analyze_tracking_status.py`](tools/codify/tracking/analyze_tracking_status.py) | Generates a summary report of AI Analysis progress from the tracking file. Shows analyzed vs pending forms for project management dashboards. |
| [`capture-code-snapshot.js`](tools/codify/utils/capture-code-snapshot.js) | Generates a single text file snapshot of code files for LLM context sharing. |
| [`export_mmd_to_image.py`](tools/codify/diagrams/export_mmd_to_image.py) | Renders all .mmd files in docs/architecture_diagrams/ to PNG images. Run this script whenever diagrams are updated to regenerate images. |
| [`generate_todo_list.py`](tools/codify/tracking/generate_todo_list.py) | Creates a prioritized TODO list of forms pending AI analysis. Bubbles up Critical and High priority items based on workflow usage. |
| [`workflow_inventory_manager.py`](tools/curate/documentation/workflow_inventory_manager.py) | Manages the workflow inventory for agent workflows (.agent/workflows/*.md). Provides search, scan, add, and update capabilities. Outputs are docs/antigravity/workflow/workflow_inventory.json and docs/antigravity/workflow/WORKFLOW_INVENTORY.md. |

## 📁 Orchestrator

| Script | Description |
| :--- | :--- |
| [`proof_check.py`](tools/orchestrator/proof_check.py) | TBD |

## 🧠 Rlm

| Script | Description |
| :--- | :--- |
| [`cleanup_cache.py`](tools/curate/rlm/cleanup_cache.py) | RLM Cleanup: Removes stale and orphan entries from the Recursive Language Model ledger. |
| [`debug_rlm.py`](tools/codify/rlm/debug_rlm.py) | Debug utility to inspect the RLMConfiguration state. Verifies path resolution, manifest loading, and environment variable overrides. Useful for troubleshooting cache path conflicts. |
| [`distiller.py`](tools/codify/rlm/distiller.py) | Recursive summarization of repo content using Ollama. |
| [`inventory.py`](tools/retrieve/rlm/inventory.py) | RLM Auditor: Reports coverage of the semantic ledger against the filesystem. Uses the Shared RLMConfig to dynamically switch between 'Legacy' (Documentation) and 'Tool' (CLI) audit modes. |
| [`query_cache.py`](tools/retrieve/rlm/query_cache.py) | RLM Search: Instant O(1) semantic search of the ledger. |
| [`rlm_config.py`](tools/codify/rlm/rlm_config.py) | Centralized configuration and utility logic for the RLM Toolchain. Implement the 'Manifest Factory' pattern to dynamically resolve manifests and cache files based on the Analysis Type (Legacy vs Tool). This module is the Single Source of Truth for RLM logic. |

## 🗄️ Vector

| Script | Description |
| :--- | :--- |
| [`cleanup.py`](tools/curate/vector/cleanup.py) | Vector Cleanup: Consistency check to remove stale chunks from DB. |
| [`ingest.py`](tools/codify/vector/ingest.py) | Vector Ingestion: Chunks code/docs and generates embeddings via ChromaDB. |
| [`ingest_code_shim.py`](tools/codify/vector/ingest_code_shim.py) | Shim for ingesting code files into Vector DB. |
| [`query.py`](tools/retrieve/vector/query.py) | Vector Search: Semantic search interface for the ChromaDB collection. |
