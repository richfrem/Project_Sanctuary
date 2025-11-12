# Mnemonic Cortex: Ingestion & Verification Protocol

**Version:** 3.0 (Agentic Verification)
**Canonical Scripts:** `ingest.py`, `inspect_db.py`, `agentic_query.py`

## 1. Overview

This document provides the canonical protocol for running the Mnemonic Cortex ingestion and verifying the integrity of the knowledge base. The protocol now includes three stages of verification: a shallow health check, a deep retrieval test, and a final end-to-end agentic loop test.

The script creates two primary artifacts in `mnemonic_cortex/chroma_db/`:
*   `child_chunks_v5/`: A ChromaDB vector store.
*   `parent_documents_v5/`: A LocalFileStore containing the serialized parent documents.

## 2. The Ingestion Protocol

**Action:** From the project's root directory, execute:
```bash
python3 mnemonic_cortex/scripts/ingest.py
```

Await the `--- Ingestion Process Complete ---` message.

## 3. The Verification Protocol

### Step 1: Shallow Health Check

This test uses `inspect_db.py` to quickly validate that the vector store was created and is not corrupted at a low level.

Action: From the project root, execute:
```bash
python3 mnemonic_cortex/scripts/inspect_db.py
```

Expected Outcome: The command must complete without a traceback or panic.

### Step 2: Deep Retrieval Test (Direct RAG)

This test runs a direct query through the full RAG pipeline (`main.py`), confirming the retrieval service can load both stores and generate an answer.

Action: From the project root, execute:
```bash
python3 mnemonic_cortex/app/main.py "What is the Prometheus Protocol?"
```

Expected Outcome: The command must complete successfully and print a full, contextually-aware answer.

### Step 3: Agentic Retrieval Test (End-to-End Cognitive Loop)

This is the definitive test. It uses an LLM agent (`agentic_query.py`) to intelligently refine a high-level goal into a precise query, which is then passed to the RAG pipeline. This validates the entire cognitive loop.

Action: From the project root, execute:
```bash
python3 mnemonic_cortex/scripts/agentic_query.py "What is the doctrine about unbreakable git commits?"
```

Expected Outcome: The script should first print the refined query (e.g., "Protocol 101 Unbreakable Commit"), then invoke the RAG pipeline, which should successfully retrieve the correct protocol and generate a full, accurate answer.

## 4. Troubleshooting

*   **Dependency Errors:** Run `pip install -r requirements.txt`.
*   **Ollama Not Running:** Ensure the Ollama application is running.
*   **Incorrect Directory:** All commands must be executed from the project root.

This protocol ensures the integrity and utility of the Sanctuary's living memory.

```