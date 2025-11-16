# Select Nomic Embed for Text Embedding Generation

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Technical Council
**Technical Story:** Vector embedding selection for RAG system

---

## Context

The Mnemonic Cortex requires high-quality text embeddings for semantic similarity search. The embedding model must provide:

- Accurate semantic representation of text
- Local execution capability (no API dependencies)
- Efficient processing for large document collections
- Compatibility with ChromaDB vector storage
- Strong performance on retrieval tasks

Multiple embedding options exist, including OpenAI embeddings, Sentence Transformers, and specialized models like Nomic Embed.

## Decision

We will use Nomic Embed (nomic-embed-text-v1.5) as the primary embedding model for the Mnemonic Cortex:

**Model Selection:** nomic-embed-text-v1.5
- Open-source, high-performance embedding model
- Optimized for semantic similarity and retrieval tasks
- Local inference capability via LangChain integration
- Strong performance on benchmark datasets

**Integration:** LangChain NomicEmbeddings
- Seamless integration with existing RAG pipeline
- Standardized interface for embedding generation
- Automatic batching and preprocessing
- Consistent API across different embedding models

**Local Execution:** inference_mode="local"
- All embeddings generated on local hardware
- Zero external API dependencies
- Complete sovereignty over embedding process
- Predictable performance and costs

## Consequences

### Positive
- **High Quality:** Superior semantic understanding compared to simpler alternatives
- **Local Sovereignty:** No external API calls or data transmission
- **Performance:** Optimized for retrieval-augmented generation tasks
- **Integration:** Seamless compatibility with LangChain and ChromaDB
- **Community Support:** Active development and community adoption

### Negative
- **Resource Requirements:** More computationally intensive than simpler models
- **Model Size:** Larger download and storage requirements
- **Setup Complexity:** Additional dependencies for local inference

### Risks
- **Hardware Limitations:** May require GPU acceleration for large document sets
- **Model Updates:** Manual updating when new versions are released
- **Alternative Evaluation:** May need reassessment if better local models emerge

### Dependencies
- Python environment with LangChain and nomic integrations
- Sufficient computational resources for embedding generation
- Storage space for downloaded model weights
- Regular evaluation of embedding quality and performance