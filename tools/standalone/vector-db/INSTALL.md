# Installation & Unpacking Guide 📂

This guide explains how to restore the "Vector DB" tool from its Markdown distribution bundle into a working directory structure.

## 0. Prerequisite: RLM Factory (Super-RAG) 🚀

This tool supports **Super-RAG**, which injects high-level summaries into code chunks.
To enable this:
1.  **Install the RLM Factory tool first.**
2.  **Run Distillation**: `python ../rlm-factory/distiller.py`
3.  **Verify Cache**: Ensure `.agent/learning/rlm_summary_cache.json` exists.

*If you skip this, Vector DB will still work but will lack semantic context injection.*

## 1. Directory Structure

Unpack the files into the following standard directory structure:

```text
<your_tool_root>/
├── vector-db/             # Tool Root
│   ├── ingest.py          # The Writer (Chunking + Embedding)
│   ├── ingest_code_shim.py# Code Parser Helper
│   ├── query.py           # The Reader (Semantic Search)
│   ├── cleanup.py         # The Janitor
│   ├── requirements.in    # Dependency Declaration
│   ├── requirements.txt   # Pinned Dependencies
│   ├── UNPACK_INSTRUCTIONS.md
│   ├── README.md
│   ├── INSTALL.md
│   ├── prompt.md
│   └── docs/
│       ├── architecture.mmd
│       └── workflow.mmd
```

## 2. Dependency Setup (Strict Policy)

This tool follows the **Dependency Management Policy** (`docs/standards/dependency_management_policy.md`).

### Python Environment
1.  **Install pip-tools** (if not present):
    ```bash
    pip install pip-tools
    ```
2.  **Compile Dependencies** (If `requirements.in` changed):
    ```bash
    pip-compile requirements.in
    ```
3.  **Sync Environment** (Safe Install):
    ```bash
    pip-sync requirements.txt
    # OR
    pip install -r requirements.txt
    ```

## 3. Configuration (.env) ⚙️

Create a `.env` file in your project root to configure the database path and RLM integration.

```bash
# --- Vector Database (ChromaDB) ---
# Used by: tools/vector-db/ingest.py, tools/vector-db/query.py
# Note: Use native Linux path (e.g. ~/.agent...) to avoid WSL/Windows I/O issues
VECTOR_DB_PATH=~/.agent/learning/chroma_db
VECTOR_DB_COLLECTION=project_forms_v1

# --- RLM Integration (Super-RAG) ---
# Required for context injection during ingestion
# Managed via: tools/standalone/rlm-factory/manifest-index.json
```

## 4. Localization (Agent / LLM Task) 🤖

**Context**: You (the Agent) have just unpacked this tool into a new location.
**Action**:
1.  **Bootstrap**: Run `pip install -r requirements.txt`.
2.  **Config**: Create the `.env` file with the variables above.
3.  **Verify**: Run `python query.py --stats` to check if it can read/create the DB.
