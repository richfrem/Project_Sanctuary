# Analysis Tools

This directory contains the core utilities for the Antigravity Command System.

> **Tool Inventory:** For a complete, auto-generated list of all scripts with their locations and descriptions, see **[`TOOL_INVENTORY.md`](TOOL_INVENTORY.md)**.

## Directory Structure

### `ai-resources/`
Centralized resources for AI/LLM assistance.
*   **`prompts/`**: System Prompts and Task Prompts.
*   **`checklists/`**: Context gathering validation lists.

### `codify/`
Tools for generating code, documentation, diagrams, and tracking progress.
*   **`documentation/`**: Overview generators and documentation tools.
*   **`diagrams/`**: Diagram generation (Mermaid export).
*   **`rlm/`**: Recursive Language Model (Intelligence Engine).
*   **`vector/`**: Embedding generation for semantic search.
*   **`tracking/`**: Task and spec tracking utilities.

### `curate/`
Tools for cleaning, organizing, and auditing the repository.
*   **`inventories/`**: Script to generate JSON/MD inventories of tools and workflows.
*   **`link-checker/`**: Utilities to find and fix broken documentation links.
*   **`documentation/`**: Workflow inventory manager.
*   **`vector/`**: Vector DB cleanup utilities.

### `investigate/`
Tools for deep exploration of the codebase.
*   **`utils/`**: Path resolution, next number generation.

### `retrieve/`
Tools for gathering context for the LLM.
*   **`bundler/`**: Creates "Smart Bundles" (single markdown files) of relevant source code.
*   **`vector/`**: Interface for querying the ChromaDB vector store.
*   **`rlm/`**: Interface for querying the RLM high-level summaries.

### `standalone/`
Self-contained tool suites with bundled documentation.
*   **`context-bundler/`**: Smart context bundling for LLM analysis.
*   **`link-checker/`**: Documentation hygiene suite.
*   **`rlm-factory/`**: RLM distillation and query tools.
*   **`vector-db/`**: ChromaDB semantic search engine.

---

## Key Workflows

### 1. Vector Database Ingestion
Ingest project files into semantic search:
```bash
python tools/codify/vector/ingest.py --full
```

### 2. RLM Distillation
Generate semantic summaries for tools and docs:
```bash
python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --all
```

### 3. Semantic Search
Query the vector database:
```bash
python tools/retrieve/vector/query.py "search term"
```

### 4. Tool Inventory
Regenerate the tool inventory:
```bash
python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py generate
```
