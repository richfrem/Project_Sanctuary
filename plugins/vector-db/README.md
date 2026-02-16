# Vector DB Plugin

**Local Semantic Search Engine powered by ChromaDB**

Search your repository by *meaning*, not just keywords. Ask "how is bail handled for youth offenders?" and get relevant code chunks ranked by semantic similarity.

## Features
- **Super-RAG**: Injects RLM file-level summaries into code chunks for richer retrieval context
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` (auto-downloaded, ~80MB)
- **Persistent Storage**: ChromaDB stores vectors locally for instant startup
- **Maintenance**: Built-in cleanup removes stale chunks from deleted files

## Prerequisites
- Python 3.8+
- `pip install chromadb sentence-transformers python-dotenv`
- Optional: `rlm-factory` plugin (for Super-RAG context injection)

## Commands

| Command | Description |
|:---|:---|
| `/vector-db:ingest` | Build or update the vector index |
| `/vector-db:query` | Semantic search across the repository |
| `/vector-db:cleanup` | Remove orphaned chunks |

## Quick Start
```bash
# 1. Build the index
python3 plugins/vector-db/scripts/ingest.py --full

# 2. Search
python3 plugins/vector-db/scripts/query.py "your question here"
```
