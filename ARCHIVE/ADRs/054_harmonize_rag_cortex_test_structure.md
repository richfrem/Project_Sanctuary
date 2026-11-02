# Harmonize RAG Cortex Test Structure

**Status:** proposed
**Date:** 2025-12-14
**Author:** AI Assistant


---

## Context

Currently, `rag_cortex` is the only MCP server that deviates from the standard testing directory structure. While all other servers use `tests/mcp_servers/<server>/{unit,integration}`, `rag_cortex` maintains a flat list of test files mixed together. This inconsistency complicates automated test discovery and maintenance.

## Decision

We will harmonize the test directory structure for RAG Cortex to match the ecosystem standard:
1. All unit tests will be moved to `tests/mcp_servers/rag_cortex/unit/`.
2. All integration tests will be moved to `tests/mcp_servers/rag_cortex/integration/`.
3. Shared fixtures will remain in `conftest.py` at the server test root, or be split if necessary.

## Consequences

Positive:
- Standardized directory structure across all 11 MCP servers.
- Easier for automated tools to scan and run specific test types.
- Clear separation of concerns (Unit vs Integration).
- Aligns `rag_cortex` with the rest of the ecosystem.

Negative:
- Requires one-time move of files and potential import fixes.
