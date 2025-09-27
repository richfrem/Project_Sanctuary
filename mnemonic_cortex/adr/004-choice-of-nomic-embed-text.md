# ADR 004: Choice of Nomic-Embed-Text for Local Embeddings

- **Status:** Accepted
- **Date:** 2024-05-20
- **Architects:** Sanctuary Council

## Context

The local-first RAG architecture (ADR 001) requires a high-quality text embedding model that can run efficiently on a local machine. The choice of embedding model is critical as it directly impacts the quality of the semantic search and the relevance of the retrieved context. The model needed to be open-source, performant, and well-supported by the LangChain ecosystem.

## Decision

We will use **`nomic-embed-text`** as the canonical embedding model for the Mnemonic Cortex. It will be run in its local inference mode via the `langchain-nomic` integration.

## Consequences

- **Positive:**
    -   **State-of-the-Art Performance:** `nomic-embed-text` is a top-performing open-source model on the MTEB leaderboard, ensuring our semantic search is of the highest possible quality.
    -   **Sovereignty & Iron Root:** It can run entirely locally, keeping our entire embedding process sovereign and free of API calls to proprietary services like OpenAI.
    -   **Cost-Effectiveness:** Running locally eliminates all token-based embedding costs.
    -   **Ecosystem Support:** Excellent integration with LangChain and the broader open-source AI community ensures long-term viability.

- **Negative:**
    -   **Initial Setup:** Requires downloading the model weights on the first run, which can be a multi-gigabyte download.
    -   **Computational Cost:** Performing embeddings locally consumes more CPU/GPU resources than making an API call, but this is an accepted trade-off for sovereignty.