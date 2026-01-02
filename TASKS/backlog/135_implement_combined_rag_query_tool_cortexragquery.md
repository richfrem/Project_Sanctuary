# TASK: Implement Combined RAG Query Tool (cortex-rag-query)

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** mcp_servers/rag_cortex/server.py, docs/architecture/mcp_gateway/guides/agent_gateway_guide.md

---

## 1. Objective

Create a single unified tool `cortex-rag-query` that combines semantic retrieval (ChromaDB) with LLM inference (Ollama) in one call, implementing the full RAG pattern for knowledge-augmented generation.

## 2. Deliverables

1. New `cortex_rag_query` function in `mcp_servers/rag_cortex/server.py`
2. Tool registration with explicit JSON schema
3. Integration with both `cortex-query` (retrieval) and `query-sanctuary-model` (generation)
4. Documentation update in agent_gateway_guide.md

## 3. Acceptance Criteria

- Tool is discoverable via Gateway /tools endpoint
- Single call retrieves relevant chunks AND generates LLM response
- Response includes both retrieved sources and generated answer
- RPC test passes: `sanctuary_cortex-cortex-rag-query`

## Notes

Pattern: 1. cortex_query() -> retrieve top-k chunks, 2. Build context prompt from chunks, 3. query_sanctuary_model() -> generate answer with context, 4. Return combined result with sources
