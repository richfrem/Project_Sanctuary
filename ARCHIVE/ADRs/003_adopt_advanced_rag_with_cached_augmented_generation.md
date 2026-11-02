# Adopt Advanced RAG with Cached Augmented Generation

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Technical Council
**Technical Story:** Mnemonic Cortex performance optimization

---

## Context

Our basic search system has significant limitations in speed and quality:

- **Fragmented Information:** Returning isolated pieces of text without full context
- **Slow Responses:** Every question requires running the full search process
- **Poor Understanding:** Simple keyword matching instead of understanding meaning
- **Wasteful Processing:** Repeating work for identical questions

Our memory system needs a more sophisticated approach to provide accurate, fast, and context-aware answers.

## Decision

We will upgrade from basic search to an advanced system that includes:

**Complete Document Retrieval:**
- Store full documents alongside search indexes
- Return complete documents instead of broken pieces
- Give the AI full context for better reasoning

**Smart Caching:**
- Use fast memory storage for query results
- Cache hit: Return instant answers for repeated questions
- Cache miss: Run full search and save the results
- Dramatically speed up common questions

**Multi-Method System:**
- Combine complete document retrieval with smart caching
- Support future advanced search capabilities
- Keep the system flexible for improvements

## Consequences

### Positive
- **Better Accuracy:** Full document context eliminates information gaps
- **Speed Improvements:** 90%+ faster responses for cached questions
- **Better Experience:** Instant answers for common questions
- **Scalability:** Efficient handling of repeated queries
- **Future-Ready:** System supports advanced search methods

### Negative
- **More Complexity:** Dual storage system (pieces + full documents)
- **Memory Usage:** In-memory cache needs RAM space
- **Cache Management:** Need ways to update and limit cache size
- **Development Time:** More complex than basic search

### Risks
- **Outdated Cache:** Old cached answers if data changes
- **Memory Pressure:** Large caches may slow down the system
- **Complexity Issues:** More parts mean more potential problems

### Dependencies
- Database collections for both search pieces and full documents
- In-memory caching system (with options to save data)
- Strategies for preparing common queries
- Monitoring for cache performance