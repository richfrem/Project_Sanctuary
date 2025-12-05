# Standardize All MCP Servers on FastMCP Implementation

**Status:** accepted
**Date:** 2025-12-05
**Author:** Antigravity Agent & Human Steward


---

## Context

During T087 Phase 2 (Comprehensive MCP Operations Testing), we discovered that the Task MCP server was using the standard MCP protocol implementation while all other 11 MCP servers (Chronicle, Protocol, ADR, Code, Config, Git, RAG Cortex, Agent Persona, Council, Orchestrator, Forge LLM) were using FastMCP.

**Background:**
- FastMCP provides a simpler, decorator-based API for defining MCP tools
- The standard MCP protocol requires async handlers and manual schema definition
- This inconsistency created:
  - Different development patterns across servers
  - Confusion when maintaining or extending MCP servers
  - Difficulty in testing operations through unified interfaces
  - Increased maintenance burden

**Problem:**
The lack of standardization violated the principle of consistency and created unnecessary complexity in the codebase. No ADR had been created to approve this architectural divergence.

## Decision

We will standardize all MCP servers on FastMCP as the canonical implementation approach.

**Rationale:**
1. **Consistency:** All 12 MCP servers should follow the same implementation pattern
2. **Simplicity:** FastMCP's `@mcp.tool()` decorator is more concise than async handlers
3. **Maintainability:** Single pattern reduces cognitive load and maintenance burden
4. **Proven:** FastMCP already successfully used by 11/12 servers
5. **Developer Experience:** Decorator-based approach is more intuitive

**Implementation:**
- Refactored Task MCP server from standard MCP to FastMCP
- Validated with comprehensive test suite (18/18 tests passing)
- Tested all 6 operations via Antigravity MCP interface
- All future MCP servers MUST use FastMCP

**Enforcement:**
- Code reviews must verify FastMCP usage for new MCP servers
- Any deviation requires explicit ADR approval
- This ADR supersedes any implicit acceptance of mixed approaches

## Consequences

**Positive:**
- Unified implementation approach across all MCP servers
- Simplified codebase maintenance and debugging
- Consistent developer experience
- Easier onboarding for new MCP server development
- Reduced cognitive load when working across multiple servers
- FastMCP's decorator-based approach is more intuitive and concise
- Automatic tool registration and schema generation
- Better error handling and validation

**Negative:**
- Required refactoring of Task MCP server (one-time cost)
- Potential breaking changes if external tools depend on old Task MCP interface
- Migration effort for any future non-FastMCP servers

**Risks:**
- Minimal - FastMCP is well-tested and used by 11/12 servers
- Task MCP refactoring validated with comprehensive test suite (18/18 tests passing)
