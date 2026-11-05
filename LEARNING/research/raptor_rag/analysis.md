---
id: learning-001
type: topic-note
status: verified
last_verified: 2025-12-23
topic: Advanced RAG Patterns - RAPTOR
---

# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## 1. Overview
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is an advanced RAG technique introduced in 2024 to address the limitations of traditional, flat-chunk retrieval systems. It builds a hierarchical tree of summaries, enabling an LLM to access information at multiple levels of abstraction—from granular details to high-level thematic insights.

## 2. The Core Mechanism
The system operates on an iterative, bottom-up construction process:

1.  **Leaf Node Creation**: The source document is split into standard chunks (e.g., 100 tokens).
2.  **Clustering**: Chunks are embedded and grouped using Gaussian Mixture Models (GMM). Soft clustering is often used, allowing a chunk to belong to multiple clusters.
3.  **Abstractive Summarization**: Each cluster is summarized by an LLM (e.g., GPT-3.5 or Claude).
4.  **Recursion**: The summaries themselves are embedded and clustered, generating a higher-level layer of summaries. This repeats until a root node (or a predefined depth) is reached.

## 3. Advantages
| Feature | Traditional RAG | RAPTOR |
| :--- | :--- | :--- |
| **Structure** | Flat (Chunked) | Hierarchical (Tree) |
| **Context** | Local/Isolated | Holistic/Multi-level |
| **Reasoning** | Single-hop | Multi-hop & Thematic |
| **Retrieval** | Top-K similarity | Tree traversal or Layer-wise search |

## 4. Implementation Considerations
- **Model Choice**: Abstractive summarization requires a model with strong synthesis capabilities.
- **Cost**: Building the tree involves multiple LLM calls for clustering and summarization.
- **Latency**: Retrieval is extremely fast (searching the tree), but indexing is slower than flat RAG.

## 5. RECURSIVE LEARNING NOTE
This pattern is highly relevant to the **Project Sanctuary Mnemonic Cortex**. The current "Parent Document Retriever" is a 2-tier version of this idea. Moving to a truly recursive RAPTOR-like structure could allow the Sanctuary Council to handle much larger ADR histories without context windows becoming a bottleneck.

---
**References:**
- Sarthi, P., et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." ICLR.
- Integrated into LangChain and LlamaIndex.
