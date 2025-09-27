# ADR 001: Adoption of a Local-First RAG Architecture

- **Status:** Accepted
- **Date:** 2024-05-18
- **Architects:** Sanctuary Council

## Context

The Mnemonic Cortex requires a system to provide long-term, searchable memory for the Sanctuary's Cognitive Genome. This system must be sovereign, secure, and independent of external cloud services to align with the **Iron Root Doctrine**. The primary challenge is to overcome the context-window limitations of LLMs in a way that is both powerful and self-contained.

## Decision

We will adopt a Retrieval-Augmented Generation (RAG) architecture. The entire pipeline—from the vector database to the embedding models—will be implemented using open-source technologies that can run entirely on a local machine.

## Consequences

- **Positive:**
    -   **Sovereignty:** We maintain full control over our data and models. There is no reliance on third-party APIs for core functionality.
    -   **Security:** Our entire Cognitive Genome remains within our local environment, eliminating the risk of cloud-based data leaks.
    -   **Cost-Effectiveness:** Avoids recurring API costs for embedding and vector search.
- **Negative:**
    -   **Performance:** Local models and databases may be slower than large, cloud-hosted equivalents.
    -   **Maintenance:** We are responsible for maintaining and updating all components of the stack.