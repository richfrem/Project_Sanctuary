# Agent Protocol: Vector DB 🧠

**Context**: You have been provided with the "Vector DB" standalone package. This is the **Semantic Search Engine**. It allows you to find valid logic patterns based on meaning, not just exact strings.

## 🤖 Your Role
You are the **Insight Miner**. Your goal is to retrieve relevant code snippets and documentation chunks that answer qualitative questions.

## 🛠️ Tool Identification
The package consists of:
- `ingest.py`: **The Builder**. Expensive operation. Reads repo, chunks it, saves vectors.
- `query.py`: **The Searcher**. Fast. Finds chunks related to your text prompt.
- `cleanup.py`: **The Janitor**. Removes ghost chunks.

## 📂 Execution Protocol

### 1. Verification (Read)
Check if the memory is healthy.
```bash
python query.py --stats
```
*   **Check**: "Status: Healthy" and "Chunks: > 0".

### 2. Retrieval (Read)
If you need to find *how* something is done:
```bash
python query.py "how do we handle youth bans"
```

### 3. Maintenance (Write)
**Only run this if**:
1.  You have changed many files.
2.  You are setting up a new environment.

```bash
# Incremental Update (Fast)
python ingest.py --since 24

# Full Rebuild (Slow - 5+ mins)
python ingest.py --full
```

## ⚠️ Critical Agent Rules
1.  **Context Injection**: Ingestion automatically reads `rlm_summary_cache.json` if configured in `.env`. This is critical for quality.
2.  **Concurrency**: Chroma is single-writer. Do not run two ingestions at once.
3.  **Persistence**: The DB lives in a folder. Do not delete that folder unless you want to re-ingest everything.
