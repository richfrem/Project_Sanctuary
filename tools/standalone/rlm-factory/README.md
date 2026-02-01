# RLM Factory 🏭

## Overview
**Recursive Language Models (RLM)** is a strategy for creating "Infinite Context" by decomposing a massive repository into individual semantic summaries stored in a searchable "Reactive Ledger".

This tool suite manages that lifecycle: **Distilling** code into summaries, **Storing** them in a JSON ledger, and **Querying** them for instant context.

Based on the research **"Recursive Language Models" (arXiv:2512.24601)**. See `research/summary.md` for the architectural theory.

For installation and unpacking instructions, see **[INSTALL.md](INSTALL.md)**.

## 🚀 Capabilities

1.  **Distill** (`distiller.py`): Recursively process files with Ollama to update the ledger.
2.  **Audit** (`inventory.py`): Check coverage of the ledger against the filesystem.
4.  **Clean** (`cleanup_cache.py`): Remove stale entries to prevent hallucinations.

## 🧠 Memory Banks (Profiles)
The RLM Factory manages distinct memory banks for different content types. Check `manifest-index.json` to select the right tool for the job.

| Type | Flag | Use for... |
| :--- | :--- | :--- |
| **Legacy Docs** | `--type legacy` | Documentation (`.md`), Business Rules. (Default) |
| **Tool Inventory** | `--type tool` | Python Scripts (`.py`), CLI Tools, Automation Logic. |

> **Tip**: If your Python script is being "skipped", you are likely running in Legacy mode by default. Switch to `--type tool`.

## ⚠️ Prerequisites

*   **Python**: 3.8+
*   **Ollama**: Exterior dependency running `granite3.2:8b`.
*   **Vector DB (Consumer)**: This tool acts as the **Producer** for the Vector DB's Super-RAG context.
    *   See `tools/standalone/vector-db/README.md` for details on how it consumes this cache.

> **🤖 Agent / LLM Note**:
> This tool requires an **Active Runtime** (Ollama). You cannot run `distiller.py` without it.
> However, `query_cache.py` and `inventory.py` are **Offline** safe—they just read the JSON file.

## Usage

### 1. Build the Ledger (The Factory)
Process all documentation to build the initial memory.
```bash
python distiller.py
```

### 2. Search Memory (The Retrieval)
Ask the ledger what a file does without reading it.
```bash
python query_cache.py "bail"
```

### 3. Maintain Hygiene (The Cleaner)
Remove dead files from memory.
```bash
python cleanup_cache.py --apply
```

## Architecture
See `architecture.mmd` for the system diagram.
