# TASK: Extend Ingestion Shim to JavaScript

**Status:** complete
**Priority:** Low
**Lead:** Unassigned
**Dependencies:** Task 110 (completed)
**Related Documents:** - mcp_servers/rag_cortex/ingest_code_shim.py\n- Task 110

---

## 1. Objective

Extend the Code Ingestion Shim to support JavaScript files, enabling the RAG system to ingest and index the frontend/Node.js codebase.

## 2. Deliverables

1. Updated ingest_code_shim.py with JS support
2. Regex-based or lightweight AST parser implementation
3. Unit/Integration tests for JS files

## 3. Acceptance Criteria

- Shim supports parsing JavaScript (.js) files
- Extracts functions, classes, and comments correctly
- Uses regex or lightweight parser (no heavy dependencies)
- Integration test validates JS ingestion
- Updated operations.py to recognize .js extensions

## Notes

**Context:**
Task 110 successfully implemented Python ingestion using `ast`.
Gemini 3.0 Pro identified that `ast` is Python-only and suggested a separate task for JavaScript to avoid scope creep and maintain the "Low-Compute" constraint.
**Strategy:**
- Do NOT use heavy AST libraries like `esprima-python` if possible (dependency bloat).
- Use a "Good Enough" Regex-based parser or a tiny Node.js sidecar script (if Node is available in environment).
- Format output as Markdown sections similar to the Python shim.
**Target Extensions:**
- .js
- .jsx (maybe)
- .ts (maybe, if regex handles types)

**Status Change (2025-12-14):** backlog → complete
Implemented Regex-based JS/TS ingestion immediately as requested by user. Validated with live integration test.
