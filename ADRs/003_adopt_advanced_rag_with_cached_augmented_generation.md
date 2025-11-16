# Adopt Advanced RAG with Cached Augmented Generation

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Technical Council
**Technical Story:** Mnemonic Cortex performance optimization

---

## Context

The basic RAG implementation suffers from significant performance and quality limitations:

- **Context Fragmentation:** Returning isolated text chunks without full document context
- **Cognitive Latency:** Every query requires full pipeline execution
- **Poor Intent Understanding:** Simple keyword matching instead of semantic understanding
- **Resource Inefficiency:** Repeated processing of identical queries

The Mnemonic Cortex requires a more sophisticated approach to provide accurate, fast, and contextually aware responses while maintaining the Doctrine of Hybrid Cognition.

## Decision

We will evolve from basic RAG to an Advanced RAG architecture incorporating:

**Parent Document Retrieval:**
- Store complete documents alongside chunked vectors
- Retrieve full documents instead of fragmented chunks
- Provide complete context to the LLM for accurate reasoning

**Cached Augmented Generation (CAG):**
- Implement high-speed in-memory caching for query results
- Cache hit: Return instant responses for repeated queries
- Cache miss: Execute full RAG pipeline and store results
- Significantly reduce latency for common queries

**Multi-Pattern Architecture:**
- Combine parent document retrieval with semantic caching
- Support future self-querying retrieval capabilities
- Maintain extensibility for advanced RAG patterns

## Consequences

### Positive
- **Improved Accuracy:** Full document context eliminates fragmentation issues
- **Performance Gains:** 90%+ faster response times for cached queries
- **Better User Experience:** Instant responses for common questions
- **Scalability:** Efficient handling of repeated queries
- **Future-Proof:** Architecture supports advanced retrieval patterns

### Negative
- **Increased Complexity:** Dual storage system (chunks + parent documents)
- **Memory Overhead:** In-memory cache requires RAM allocation
- **Cache Management:** Need strategies for cache invalidation and size limits
- **Development Time:** More complex implementation than basic RAG

### Risks
- **Cache Staleness:** Outdated cached responses if underlying data changes
- **Memory Pressure:** Large cache sizes may impact system performance
- **Complexity Overhead:** Additional moving parts increase failure points

### Dependencies
- ChromaDB collections for both chunked vectors and parent documents
- In-memory caching mechanism (Python dict with persistence options)
- Cache warming strategies for common queries
- Monitoring and metrics for cache hit/miss ratios