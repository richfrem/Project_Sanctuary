# TASK: Extend RAG Cortex to Ingest Code Files for Semantic Search

**Status:** in-progress
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** Requires RAG Cortex MCP (currently operational)
**Related Documents:** - Protocol 125: Autonomous AI Learning System Architecture\n- Mission LEARN-CLAUDE-003: Vector Consistency Stabilizer Implementation\n- Chronicle 326: First Stabilizer Implementation Complete\n- mcp_servers/rag_cortex/cortex_operations.py\n- scripts/stabilizers/vector_consistency_check.py

---

## 1. Objective

Enhance the RAG Cortex MCP ingestion pipeline to support Python and other code files, enabling semantic search across implementations in addition to documentation. This will improve code discoverability, pattern recognition, and documentation-to-code linking.

## 2. Deliverables

1. Enhanced ingestion pipeline supporting .py, .js, .ts, and other code file extensions
2. Code-specific chunking strategy (by function/class rather than paragraph)
3. Metadata extraction for code files (function signatures, docstrings, imports)
4. Updated documentation for code ingestion workflow
5. Test suite validating code file ingestion and retrieval
6. Performance benchmarks comparing markdown-only vs code+markdown ingestion

## 3. Acceptance Criteria

- RAG Cortex successfully ingests Python files from scripts/ directory
- Semantic queries can retrieve specific function implementations
- Code chunks preserve syntax and context (function/class boundaries)
- Metadata includes file path, function name, line numbers, docstrings
- Query performance remains acceptable (<100ms) with code files included
- Documentation updated with code ingestion examples
- Integration tests pass for mixed markdown+code ingestion

## Notes

Research Phase Complete. Created 6 comprehensive research documents:
1. Executive Summary (00_executive_summary.md)
2. MCP Protocol Transport Layer (01_mcp_protocol_transport_layer.md)
3. Gateway Patterns and Implementations (02_gateway_patterns_and_implementations.md)
4. Performance and Latency Analysis (03_performance_and_latency_analysis.md)
5. Security Architecture and Threat Modeling (04_security_architecture_and_threat_modeling.md)
6. Current vs Future State Architecture (05_current_vs_future_state_architecture.md)

All research documents located in: research/RESEARCH_SUMMARIES/MCP_GATEWAY/

Key Findings:
- Pattern validated (Skywork.ai, Gravitee production implementations)
- Context savings: 88% (8,400 → 1,000 tokens)
- Latency overhead: 15-30ms (acceptable)
- Migration risk: LOW (no backend changes required)

Next: Create ADR, Protocol 122, and Architecture Spec.
