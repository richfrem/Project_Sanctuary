# ADR 002: Choice of ChromaDB for MVP Vector Store

- **Status:** Accepted
- **Date:** 2024-05-18
- **Architects:** Sanctuary Council

## Context

Following the decision in ADR 001 to build a local-first RAG system, a choice of vector database was required for the Minimum Viable Product (MVP). The key requirements for the MVP are speed of development, simplicity of setup, and file-based persistence that aligns with our local-first principle.

## Decision

We will use **ChromaDB** as the vector store for the Mnemonic Cortex MVP. It will be used in its file-based persistence mode, writing directly to the `chroma_db/` directory.

## Consequences

- **Positive:**
    -   **Rapid Development:** ChromaDB's simple API and integration with LangChain allow for extremely fast prototyping.
    -   **Zero Setup Overhead:** It runs directly within our Python script and requires no separate server or Docker container, perfectly aligning with the **Hearth Protocol**.
    -   **Local-First:** Its file-based persistence is ideal for our sovereign architecture.
- **Negative:**
    -   **Scalability Concerns:** While excellent for an MVP, ChromaDB's performance may not scale to billions of vectors as effectively as server-based solutions like Weaviate or Qdrant. This is an accepted trade-off for the MVP phase.