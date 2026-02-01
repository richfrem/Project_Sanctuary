# Installation & Unpacking Guide 📂

This guide explains how to restore the "RLM Factory" tool from its Markdown distribution bundle into a working directory structure.

## 1. Directory Structure

For optimal usage by an Agent/LLM or in a clean environment, unpack the files into the following standard directory structure:

```text
<your_tool_root>/
├── rlm-factory/           # Tool Root
│   ├── distiller.py       # The Engine (Write)
│   ├── inventory.py       # The Auditor (Read)
│   ├── cleanup_cache.py   # The Janitor (Curate)
│   ├── query_cache.py     # The Interface (Search)
│   ├── requirements.in    # Dependency Declaration
│   ├── requirements.txt   # Pinned Dependencies
│   ├── UNPACK_INSTRUCTIONS.md
│   ├── README.md
│   ├── INSTALL.md
│   ├── prompt.md
│   ├── research/
│   │   └── summary.md     # Architectural Context
│   └── docs/
│       ├── architecture.mmd
│       └── workflow.mmd
```

## 2. Dependency Setup (Strict Policy)

This tool follows the **Dependency Management Policy** (`docs/standards/dependency_management_policy.md`). We use `pip-tools` to ensure reproducible builds.

### Prerequisite: Ollama (The Brain)
The RLM relies on a local LLM server.
1.  **Install Ollama**: [https://ollama.com/](https://ollama.com/)
2.  **Pull Model**: We strictly use the optimized `granite3.2:8b` model.
    ```bash
    ollama pull granite3.2:8b
    ```
3.  **Start Server**:
    ```bash
    ollama serve
    ```

### Python Environment
1.  **Install pip-tools** (if not present):
    ```bash
    pip install pip-tools
    ```
2.  **Compile Dependencies** (If `requirements.in` changed):
    ```bash
    pip-compile requirements.in
    ```
3.  **Sync Environment** (The only safe install method):
    ```bash
    pip-sync requirements.txt
    # OR standard pip install
    pip install -r requirements.txt
    ```

## 3. Configuration (.env)

Create a `.env` file in your project root to configure the factory.

```bash
# --- Ollama Configuration (Model Serving) ---
# Used by: RLM Distiller
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=granite3.2:8b

# --- RLM Factory (Recursive Language Model) ---
# Used by: distiller.py
# CACHE PATH is now managed via tools/standalone/rlm-factory/manifest-index.json
```

## 4. Localization (Agent / LLM Task) 🤖

**Context**: You (the Agent) have just unpacked this tool into a new location.
**Action**:
1.  **Bootstrap**: Run `pip install -r requirements.txt`.
2.  **Config**: Create the `.env` file.
3.  **Verify**: Run `python inventory.py` - it will automatically resolve the cache path via `manifest-index.json`.
