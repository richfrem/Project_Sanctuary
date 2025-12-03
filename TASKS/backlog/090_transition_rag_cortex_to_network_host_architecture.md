# TASK: Transition RAG Cortex to Network Host Architecture

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Transition RAG Cortex to Network Host Architecture

## 2. Deliverables

1. .vector_data directory
2. Updated .env
3. Updated README.md

## 3. Acceptance Criteria

- .vector_data directory exists
- .env file updated with CHROMA_HOST and CHROMA_PORT
- docs/mcp/servers/rag_cortex/README.md updated with Architectural Shift and Ingest Protocol
- docker-compose.yml uses bind mount for vector-db (if applicable)
