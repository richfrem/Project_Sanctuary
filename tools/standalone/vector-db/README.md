# Vector DB (Chroma) 🗄️

## Overview
A persistent, local Vector Database powered by **ChromaDB**. It provides semantic search capabilities for the repository, enabling "Concept-based" retrieval (e.g., "Find logic about bail") rather than just keyword matching.

Features **Super-RAG**: Integrates with the **RLM Factory** to inject file-level summaries into code chunks for superior retrieval context.

For installation and unpacking instructions, see **[INSTALL.md](INSTALL.md)**.

## 🚀 Capabilities

1.  **Ingest** (`ingest.py`): Chunk files, inject RLM context, embed using HuggingFace (`all-MiniLM-L6-v2`), and store in Chroma.
2.  **Query** (`query.py`): Perform cosine similarity search to find relevant docs/code.
3.  **Clean** (`cleanup.py`): Remove stale chunks to prevent hallucinations.

## ⚠️ Prerequisites

*   **Python**: 3.8+
*   **Disk**: Local storage for Chroma DB files.
*   **RAM**: Enough to load the embedding model (~500MB).
*   **RLM Factory (Producer)**: Install and run `rlm-factory` first.
    *   This tool acts as the **Consumer** of the RLM cache for Super-RAG context injection.

> **🤖 Agent / LLM Note**:
> This tool creates persistent binary files in `VECTOR_DB_PATH`. Do not try to read those files directly. Always use `query.py`.

## Usage

### 1. Ingest (Build Memory)
Capture the current state of documentation and code.
```bash
python ingest.py --full
```
*Tip: Ensure RLM Cache exists first for better quality.*

### 2. Search (Retrieve Memory)
Ask the database a conceptual question.
```bash
python query.py "how is checking bail different from granting it?"
```

### 3. Maintain (Cleanup)
Remove chunks from deleted files.
```bash
python cleanup.py --apply
```
