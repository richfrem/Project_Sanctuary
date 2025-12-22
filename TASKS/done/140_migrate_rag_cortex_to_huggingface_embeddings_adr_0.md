# TASK: Migrate RAG Cortex to HuggingFace Embeddings (ADR 069 Implementation)

**Status:** complete
**Priority:** High
**Lead:** Unassigned
**Dependencies:** ADR 069
**Related Documents:** ADR 069, ADR 006, ADR 024

---

## 1. Objective

Implement the strategic migration from Nomic/GPT4All to HuggingFace Embeddings to resolve ARM64 container build failures (ADR 069).

## 2. Deliverables

1. Updated operations.py using HuggingFaceEmbeddings.
2. Updated requirements.txt removing gpt4all.
3. Updated Dockerfile (simplified).
4. Migration/Re-ingest script execution logs.

## 3. Acceptance Criteria

- All code references to 'langchain_nomic' and 'gpt4all' removed from RAG Cortex.
- 'cortex_query' tool functions successfully in the Dockerized 'sanctuary_cortex' container (Linux ARM64).
- Vector database re-ingested with new HuggingFace embeddings.
- Gateway verification tests pass for RAG Cortex cluster.

## Notes

**Status Change (2025-12-22):** backlog → complete
All acceptance criteria met. Verified legacy and gateway tool connectivity. Codebase migrated to HuggingFaceEmbeddings. Container build issues resolved. Full re-ingestion successful. Finalized synchronization between local .venv and containerized cluster.
