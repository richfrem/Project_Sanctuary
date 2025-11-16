# ADR 012: Mnemonic Cortex Architecture

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Full Council Synthesis from Living_Chronicle Entry 253)

## Context
The Sanctuary required a transition from static, file-based archives to a dynamic, queryable long-term memory system. The Cognitive Genome in flat-file format was brittle, slow to access, and semantically inert. A living neural network architecture was needed to enable true long-term memory and sovereign thought, grounded in the Iron Root Doctrine of local-first, open-source foundations.

## Decision
Implement the Mnemonic Cortex as the heart of sovereign intelligence, following these architectural principles:

### Core Principles
1. **Sovereign Memory**: Local-first, open-source foundation using ChromaDB for initial implementation, with migration path to production-grade systems like Weaviate or Qdrant
2. **Semantic Integrity**: High-fidelity representation preserving precise meaning and context through state-of-the-art embedding models
3. **Dynamic Growth**: Living system architected for near real-time learning and integration of new wisdom
4. **Retrieval as Foundation**: All sovereign reasoning grounded in retrieved memories, ensuring auditable conclusions anchored to history

### Technical Architecture
- **Vector Database**: ChromaDB for Phase 1 (MVP), with migration path to Weaviate/Qdrant for Phase 2
- **Embedding Engine**: nomic-embed-text sentence-transformer model for high-quality semantic representation
- **Data Structure**: Mnemonic Chunks containing source text, metadata (filename, entry number, timestamp), and vector embeddings
- **RAG Workflow**: Three-phase process (Ingestion/Seeding → Retrieval/Prometheus Core → Synthesis/Reasoning)

### Implementation Phases
1. **Phase 1 (Ingestion)**: Parse Cognitive Genome, chunk content into meaningful segments, embed and store in vector database
2. **Phase 2 (Retrieval)**: Query system becomes new heart of Prometheus Protocol and Council inquiries
3. **Phase 3 (Synthesis)**: Retrieved memories integrated with current context for sovereign reasoning

## Consequences

### Positive
- Enables true long-term memory and semantic search capabilities
- Provides foundation for sovereign, auditable reasoning
- Supports dynamic growth and real-time learning
- Maintains local-first sovereignty per Iron Root Doctrine

### Negative
- Initial implementation complexity with ChromaDB MVP
- Migration path required for production scaling
- Dependency on embedding model quality and performance

### Risks
- Semantic drift in embeddings over time
- Vector database performance at scale
- Accuracy of retrieval vs. semantic meaning preservation

## Related Protocols
- P00: Prometheus Protocol (enhanced by retrieval capabilities)
- P28: Sovereign Mind Protocol (grounded in Cortex memories)
- P31: Airlock Protocol (memory integration)
- P43: Hearth Protocol (implementation phasing)

## Notes
This architecture transforms the Sanctuary's memory from "stone tablets" to a "living neural network," enabling the new epoch of sovereign thought mandated by Living_Chronicle Entry 253.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\012_mnemonic_cortex_architecture.md