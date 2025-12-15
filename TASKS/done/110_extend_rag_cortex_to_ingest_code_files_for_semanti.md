# TASK: Extend RAG Cortex to Ingest Code Files for Semantic Search

**Status:** complete
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

**Context:**
This enhancement was identified during Mission LEARN-CLAUDE-003 when attempting to ingest the Vector Consistency Stabilizer Python implementation. Currently, RAG Cortex only supports markdown files, which limits semantic search to documentation.
**STRATEGIC BREAKTHROUGH (Gemini 3.0 Pro):**
Instead of rewriting RAG Cortex to understand code natively, use an **AST-based "Pseudo-Markdown" Converter** that transforms Python code into markdown format the Cortex already understands.
**Low-Compute Strategy:**
1. **Zero Tokens:** Use Python's built-in `ast` module (millisecond parsing)
2. **No Agents:** Hard-coded logic, no orchestrator needed
3. **Existing Infrastructure:** Feeds into current `cortex_ingest_incremental`
4. **Fast:** <100ms per file
**Implementation Approach:**
**Step 1:** Create `scripts/ingest_code_shim.py`
- Parse Python files using `ast` module
- Extract functions/classes as markdown sections
- Include metadata: line numbers, docstrings, signatures
- Format as markdown with code blocks
**Step 2:** Convert code to "virtual markdown"
```python
def parse_python_to_markdown(file_path):
    # Use ast.parse() to get syntax tree
    # Extract module docstring as ## Module Description
    # For each function/class:
    #   - Create ## Function: name section
    #   - Include line number, docstring
    #   - Embed source code in ```python block
    # Return markdown string
```
**Step 3:** Ingest via existing pipeline
```python
virtual_markdown = parse_python_to_markdown('vector_consistency_check.py')
cortex_ingest_incremental(content=virtual_markdown, path=file_path + '.md')
```
**Benefits:**
- ✅ Chunking by function/class (not paragraphs)
- ✅ Metadata extraction (docstrings, signatures, line numbers)
- ✅ Zero LLM tokens for parsing
- ✅ Works with existing RAG infrastructure
- ✅ Fast (<100ms per file)
**Technical Considerations:**
1. **AST Parsing:** Use `ast.parse()`, `ast.get_docstring()`, `ast.get_source_segment()`
2. **Chunking Strategy:** One markdown section per function/class
3. **Metadata:** Extract line numbers, function signatures, docstrings
4. **Language Support:** Start with Python, expand to JS/TS later
5. **Performance:** Monitor vector DB size and query latency
**Success Metrics:**
- Code files successfully ingested and retrievable
- Query: 'How does vector_consistency_check work?' returns the actual function
- Performance: <100ms conversion + ingestion time
- Accuracy: Relevant code chunks in top 3 results
**Related Work:**
- Code MCP already has AST utilities that could be leveraged
- Python's `ast` module is built-in (no dependencies)
- Consider extending to JavaScript/TypeScript using similar parsers

**Status Change (2025-12-14):** backlog → complete
Implemented `scripts/ingest_code_shim.py` which uses Python AST to parse code files into a markdown format that RAG Cortex can ingest. Validated by ingesting `vector_consistency_check.py` and successfully querying its function logic. Zero-token overhead strategy used successfully.
