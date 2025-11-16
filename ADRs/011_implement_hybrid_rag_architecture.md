# Implement Hybrid RAG Architecture with Multi-Pattern Integration

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Sanctuary Council
**Technical Story:** Mnemonic Cortex architecture evolution

---

## Context

The Mnemonic Cortex began with basic RAG implementation but evolved to address critical limitations in retrieval quality, context preservation, and performance. Basic RAG suffered from:

- **Context Fragmentation:** Isolated text chunks lacked full document context
- **Cognitive Latency:** Every query required complete pipeline execution
- **Poor Intent Understanding:** Simple semantic search missed nuanced query requirements
- **Resource Inefficiency:** Repeated processing of identical queries

The system required a hybrid approach combining multiple advanced RAG patterns to create a sophisticated, multi-tiered retrieval system that is fast, accurate, and contextually aware.

## Decision

We will implement a hybrid RAG architecture that integrates three complementary advanced retrieval patterns:

**Parent Document Retrieval + Dual Collection Storage:**
- **Child Collection:** Semantic chunks with vector embeddings for similarity search
- **Parent Collection:** Complete documents stored separately for full context retrieval
- **Retrieval Logic:** Find relevant chunks → Return associated parent documents
- **Benefits:** Preserves document integrity while maintaining efficient search

**Self-Querying Retrieval with Structured Query Generation:**
- **Query Analysis:** LLM parses natural language queries to extract intent and constraints
- **Structured Output:** Generates JSON with semantic queries, metadata filters, and search parameters
- **Enhanced Precision:** Supports complex queries with temporal, authority, and content filters
- **Benefits:** Transforms retrieval from keyword matching to intelligent understanding

**Cached Augmented Generation (CAG) with Multi-Tier Caching:**
- **Hot Cache:** In-memory Python dict for sub-millisecond responses
- **Warm Cache:** SQLite persistence for cross-session availability
- **Query Fingerprinting:** SHA-256 hash of query + model + knowledge base version
- **Benefits:** 90%+ performance improvement for repeated queries

## Consequences

### Positive
- **Superior Accuracy:** Full document context eliminates fragmentation issues
- **Intelligent Retrieval:** Self-querying understands complex query requirements
- **Performance Optimization:** Caching provides instant responses for common queries
- **Scalability:** Multi-tier architecture handles varying query patterns efficiently
- **Future-Proof:** Modular design supports additional retrieval patterns

### Negative
- **Architectural Complexity:** Three interdependent systems require careful coordination
- **Resource Overhead:** Dual storage and caching increase memory requirements
- **Development Complexity:** Multiple moving parts complicate testing and debugging
- **Maintenance Burden:** Each pattern requires separate optimization and monitoring

### Risks
- **Integration Challenges:** Patterns must work seamlessly together
- **Performance Bottlenecks:** Cache misses still require full pipeline execution
- **Data Consistency:** Dual collections must remain synchronized
- **Query Planning Overhead:** Self-querying adds latency for simple queries

### Dependencies
- ChromaDB dual collection setup (chunks + parent documents)
- LLM integration for self-querying capabilities
- In-memory + SQLite caching infrastructure
- Query fingerprinting and cache key generation
- Performance monitoring for cache hit/miss ratios and retrieval accuracy