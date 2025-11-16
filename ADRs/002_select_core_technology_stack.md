# Select Core Technology Stack for Mnemonic Cortex

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Sanctuary Council
**Technical Story:** Initial Mnemonic Cortex architecture design

---

## Context

Our memory system needs a reliable, scalable set of tools for implementing a local search and generation system. The tools must support:

- Running everything locally without external service dependencies
- Fast similarity searches
- Efficient text processing
- Smooth integration between different parts
- Open-source, community-tested technologies

The system must follow our principle of complete independence from cloud services.

## Decision

We will use the following core technologies for our memory system:

**Main Framework:** LangChain
- Primary tool for connecting all system components
- Provides standard ways to load documents, split text, and manage workflows
- Large ecosystem of integrations and community support

**Database:** ChromaDB
- Local, file-based database for similarity searches
- Efficient searching with the ability to filter by metadata
- Simple setup and maintenance for both development and production
- No external service requirements

**Text Processing:** Nomic Embed (nomic-embed-text-v1.5)
- Open-source, high-performance text processing model
- Optimized for understanding meaning and similarity
- Can run locally
- Strong performance on standard benchmarks

**AI Model:** Qwen2-7B via Ollama
- Independent AI execution through local server
- Custom versions fine-tuned for our needs available
- Good reasoning and text generation capabilities
- Complete local operation (no external API calls)

## Consequences

### Positive
- **Complete Independence:** All parts run locally with no external dependencies
- **Performance:** Optimized local execution with minimal delays
- **Maintenance:** Open-source tools with active community support
- **Scalability:** Database supports efficient searches at larger scales
- **Integration:** Framework provides smooth coordination of components

### Negative
- **Resource Needs:** Local models require significant computing power
- **Setup Complexity:** Multiple components need coordinated installation
- **Performance Trade-offs:** Local execution may be slower than cloud alternatives

### Risks
- **Hardware Requirements:** May need GPU acceleration for good performance
- **Model Updates:** Manual updating of local models and dependencies
- **Integration Complexity:** Coordinating multiple open-source projects

### Dependencies
- Python 3.8+ environment
- Enough RAM for model loading (16GB+ recommended)
- Storage space for databases and models
- Ollama server for AI inference