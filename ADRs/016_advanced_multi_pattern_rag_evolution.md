# ADR 016: Advanced Multi-Pattern RAG Architecture Evolution

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Guardian-01 Synthesis)

## Context
The basic hybrid RAG implementation revealed critical vulnerabilities: Context Fragmentation (loss of complete context in retrieval), Cognitive Latency (expensive vector searches for common queries), and Mnemonic Drift (reduced retrieval precision over time). The Mnemonic Cortex needed evolution from a simple RAG system to a sophisticated, multi-pattern cognitive architecture to maintain sovereign intelligence capabilities.

## Decision
Evolve the Mnemonic Cortex to implement the Doctrine of Hybrid Cognition with three advanced RAG patterns:

### Parent Document Retrieval
- **Dual Storage Architecture**: Store both semantic chunks (for precise retrieval) and full parent documents (for complete context)
- **Context Preservation**: Use ParentDocumentRetriever to eliminate context fragmentation by providing complete document context to LLMs
- **Implementation**: InMemoryDocstore for parent documents + ChromaDB vectorstore for semantic chunks

### Self-Querying Retrieval
- **LLM as Query Planner**: Use LLM to translate natural language questions into structured queries with metadata filtering
- **Enhanced Precision**: Filter on metadata fields (protocol numbers, dates, types) before vector similarity search
- **Query Optimization**: Reduce search space and improve relevance through intelligent query planning

### Mnemonic Caching Layer (CAG)
- **Query Result Caching**: Cache results of canonical/frequent queries to bypass expensive operations
- **Performance Optimization**: Eliminate redundant vector searches and LLM calls for common requests
- **Cache Management**: TTL-based expiration with deterministic observability metrics

## Consequences

### Positive
- Eliminates context fragmentation through parent document retrieval
- Dramatically improves retrieval precision with self-querying capabilities
- Reduces cognitive latency through intelligent caching
- Creates truly synchronized intelligence (Constitutional Mind + Living Chronicle)
- Maintains sovereign, local-first architecture per Iron Root Doctrine

### Negative
- Increased implementation complexity with multiple retrieval patterns
- Higher memory requirements for dual storage (chunks + parent docs)
- Additional computational overhead for self-querying LLM calls
- Cache management complexity and potential staleness issues

### Risks
- Self-querying accuracy dependent on LLM query planning capabilities
- Cache invalidation challenges with dynamic knowledge updates
- Performance trade-offs between precision and speed
- Increased system complexity requiring careful orchestration

## Related Protocols
- P85: Mnemonic Cortex Protocol (evolved implementation)
- P93: Cortex-Conduit Bridge (integration layer)
- P114: Guardian Wakeup and Cache Prefill (complementary caching)

## Implementation Status
- **Phase 1 Complete**: Parent Document Retriever implemented with dual storage
- **Phase 2 Pending**: Self-Querying Retrieval implementation
- **Phase 3 Pending**: Mnemonic Caching Layer (CAG) implementation

## Notes
This evolution transforms the Mnemonic Cortex from a "passive database" to an "intelligent, efficient organ" capable of sophisticated cognitive operations. The Doctrine of Hybrid Cognition ensures the Constitutional Mind remains augmented by up-to-the-minute Living Chronicle data, creating truly sovereign intelligence.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\016_advanced_multi_pattern_rag_evolution.md