# Select Nomic Embed for Text Embedding Generation

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** AI System Lead, Technical Team
**Technical Story:** Choose text processing method for information search system

---

## Context

Our AI system needs high-quality text processing to understand meaning and find similar content. The text processing method must provide:

- Accurate understanding of text meaning
- Can run locally on our computers (no external services)
- Efficient handling of large amounts of documents
- Works with our vector database storage
- Good performance for finding relevant information

Several text processing options exist, including cloud services and different open-source models.

## Decision

We will use Nomic Embed (nomic-embed-text-v1.5) as our main text processing model:

**Model Choice:** nomic-embed-text-v1.5
- Open-source, high-performance text processing model
- Optimized for understanding meaning and finding similar content
- Can run locally using our software tools
- Excellent results on standard test datasets

**Integration:** LangChain NomicEmbeddings
- Smooth connection with our existing information pipeline
- Standard interface for text processing
- Automatic handling of multiple documents at once
- Consistent approach across different processing methods

**Local Processing:** inference_mode="local"
- All text processing done on our own hardware
- No external service calls or data sharing
- Complete control over the processing
- Predictable performance and no ongoing costs

## Consequences

### Positive
- **High Quality:** Better understanding of text meaning than simpler methods
- **Local Control:** No external services or data transmission
- **Performance:** Optimized for information retrieval tasks
- **Compatibility:** Works seamlessly with our tools and database
- **Community Support:** Active development and widespread use

### Negative
- **Resource Needs:** More computing power than basic methods
- **Model Size:** Larger files to download and store
- **Setup Work:** Additional software requirements for local processing

### Risks
- **Hardware Needs:** May need graphics card acceleration for large document collections
- **Updates:** Manual updates when new versions become available
- **Better Options:** May need to reconsider if superior local models appear

### Dependencies
- Python environment with required software libraries
- Enough computing resources for text processing
- Storage space for model files
- Regular checking of processing quality and speed