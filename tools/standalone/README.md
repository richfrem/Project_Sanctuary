# Standalone Tool Registry

This directory contains self-contained tool suites ("Bubbles") designed for specific tasks within the Project Sanctuary pipeline. Each tool is bundled with its own logic, dependencies, and documentation, allowing it to function as an independent agent capability.

## 📦 Active Tool Suites

### 1. [Context Bundler](./context-bundler/)
**Purpose:** Creates "Smart Bundles" (single Markdown artifacts) from scattered source files for LLM analysis.
**Key Components:** `manifest_manager.py`, `bundle.py`.
**Use Case:** "I need to give the AI context about multiple related files."

### 2. [Link Checker](./link-checker/)
**Purpose:** Documentation hygiene suite that indexes the repo, finds broken links, and auto-corrects them using fuzzy matching.
**Key Components:** `check_broken_paths.py` (Inspector), `smart_fix_links.py` (Fixer), `map_repository_files.py` (Mapper).
**Use Case:** "Fix all broken links in the documentation."

### 3. [RLM Factory](./rlm-factory/)
**Purpose:** The engine behind **Recursive Language Models**. Distills code into semantic ledger entries for O(1) context retrieval.
**Key Components:** `distiller.py` (Producer), `query_cache.py` (Consumer), `inventory.py` (Auditor).
**Use Case:** "Summarize this large file so I can search its logic instantly."

### 4. [Vector DB](./vector-db/)
**Purpose:** Local semantic search engine powered by ChromaDB, enabling concept-based retrieval.
**Key Components:** `ingest.py`, `query.py`, `cleanup.py`.
**Use Case:** "Find all code related to a specific concept across the entire repository."

---

## 🛠️ Integration Note
While these tools reside in `standalone/` for modularity, they are fully integrated into the main **Antigravity CLI** (`tools/cli.py`). The CLI orchestrates these tools to perform complex workflows.
