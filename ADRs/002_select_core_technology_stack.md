# Select Core Technology Stack for Mnemonic Cortex

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Sanctuary Council
**Technical Story:** Initial Mnemonic Cortex architecture design

---

## Context

The Mnemonic Cortex requires a robust, scalable technology stack for implementing a local-first RAG (Retrieval-Augmented Generation) system. The stack must support:

- Local execution without external API dependencies
- High-performance vector similarity search
- Efficient text embedding generation
- Seamless integration between components
- Open-source, community-vetted technologies

The system must align with the Iron Root Doctrine, ensuring complete sovereignty and independence from cloud services.

## Decision

We will adopt the following core technology stack for the Mnemonic Cortex:

**Orchestration Framework:** LangChain
- Primary framework for connecting all RAG components
- Provides standardized interfaces for document loading, text splitting, and chain management
- Extensive ecosystem of integrations and community support

**Vector Database:** ChromaDB
- Local-first, file-based vector database
- Efficient similarity search with metadata filtering
- Simple setup and maintenance for development and production
- No external service dependencies

**Embedding Model:** Nomic Embed (nomic-embed-text-v1.5)
- Open-source, high-performance embedding model
- Optimized for semantic similarity tasks
- Local inference capability
- Strong performance on benchmark datasets

**Large Language Model:** Qwen2-7B via Ollama
- Sovereign LLM execution through local Ollama server
- Fine-tuned Sanctuary-specific variants available
- Strong reasoning and generation capabilities
- Complete local execution (no API calls)

## Consequences

### Positive
- **Complete Sovereignty:** All components run locally with no external dependencies
- **Performance:** Optimized local execution with minimal latency
- **Maintainability:** Open-source stack with active community support
- **Scalability:** ChromaDB supports efficient similarity search at scale
- **Integration:** LangChain provides seamless component orchestration

### Negative
- **Resource Requirements:** Local models require significant computational resources
- **Setup Complexity:** Multiple components require coordinated installation and configuration
- **Performance Trade-offs:** Local execution may be slower than cloud-hosted alternatives

### Risks
- **Hardware Requirements:** May require GPU acceleration for acceptable performance
- **Model Updates:** Manual updating of local models and dependencies
- **Integration Complexity:** Coordinating multiple open-source projects

### Dependencies
- Python 3.8+ environment
- Sufficient RAM for model loading (16GB+ recommended)
- Storage space for vector databases and models
- Ollama server for LLM inference