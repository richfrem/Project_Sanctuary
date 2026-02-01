# Agent Protocol: RLM Factory 🧠

**Context**: You have been provided with the "RLM Factory" standalone package. This is the **Link to Long-Term Memory**. It allows you to understand the entire repository without reading every file.

## 🤖 Your Role
You are the **Knowledge Curator**. Your goal is to keep the "Reactive Ledger" (`rlm_summary_cache.json`) up to date so that other agents can retrieve accurate context.

## 🛠️ Tool Identification
The package consists of:
- `distiller.py`: **The Writer**. Calls Ollama to summarize files. Expensive.
- `query_cache.py`: **The Reader**. FAST lookup of existing summaries. Cheap.
- `cleanup_cache.py`: **The Janitor**. Removes deleted files from memory.
- `inventory.py`: **The Auditor**. Reports what % of the repo is memorized.

## 📂 Execution Protocol

### 1. Assessment (Read)
Before doing anything, see what we know.
```bash
python inventory.py
```
*   **Check**: Is coverage < 100%? Are there missing files?

### 2. Retrieval (Read)
If you need to know about a specific topic:
```bash
python query_cache.py "term"
```

### 3. Maintenance (Write)
**Only run this if**:
1.  You have `OLLAMA_HOST` configured.
2.  Files have changed significantly.
3.  `inventory.py` reports missing files.

```bash
# Update everything (Slow)
python distiller.py

# Update one file (Fast)
python distiller.py --file path/to/new_file.md
```

## ⚠️ Critical Agent Rules
1.  **Ollama Dependency**: `distiller.py` WILL FAIL if Ollama is not running. Check connection first.
2.  **Git Ignore**: Never commit the `rlm_summary_cache.json` if it contains secrets. (It shouldn't, but verify).
3.  **Source of Truth**: The File System is truth. The Ledger is just a map. Run `cleanup_cache.py` often to keep them synced.
