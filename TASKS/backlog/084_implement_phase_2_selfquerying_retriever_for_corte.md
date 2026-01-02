# TASK: Implement Phase 2: Self-Querying Retriever for Cortex MCP

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** docs/architecture/mcp/cortex_evolution.md, docs/architecture/mcp/RAG_STRATEGIES.md, ARCHIVE/mnemonic_cortex/app/services/llm_service.py

---

## 1. Objective

Implement Phase 2 of Cortex evolution (Self-Querying Retriever) to enable LLM-powered query decomposition and metadata filtering. This will allow natural language queries to be automatically structured with filters and optimized for vector search.

## 2. Deliverables

1. Enhanced cortex_query() tool with self-querying capability
2. Query decomposition logic from llm_service.py integrated
3. Metadata filtering support (source, protocol number, date ranges)
4. Structured query generation with LLM
5. Tests for self-querying functionality
6. Documentation of Phase 2 implementation

## 3. Acceptance Criteria

- cortex_query() can decompose complex queries into sub-queries
- Metadata filters are automatically extracted from natural language
- Query optimization improves retrieval accuracy
- LLM-powered query structuring is functional
- Tests pass for self-querying scenarios
- Documentation updated with Phase 2 features

## Notes

Reference implementation in ARCHIVE/mnemonic_cortex/app/services/llm_service.py after archival. This is Phase 2 of the Cortex evolution plan (see docs/architecture/mcp/cortex_evolution.md). Requires LLM integration for query structuring and JSON output parsing.
