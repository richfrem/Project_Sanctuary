# MCP Server Separation of Concerns

**Status:** accepted
**Date:** 2025-11-30
**Author:** Antigravity (Council MCP Implementation)


---

## Context

As the Project Sanctuary MCP ecosystem evolved, we implemented multiple specialized MCP servers (Code, Git, Cortex, Protocol, Task, Chronicle, ADR, Config). When implementing the Council MCP Server, we faced a design decision: should the Council MCP duplicate functionality from other MCPs (file I/O, git operations, memory queries) for convenience, or should it focus solely on its unique capability (multi-agent deliberation) and compose with other MCPs?

The Council Orchestrator historically had built-in file I/O, git operations, and memory access. The question was whether the MCP wrapper should expose all these capabilities or delegate to specialized MCPs.

## Decision

We will enforce strict separation of concerns across all MCP servers. Each MCP server provides ONLY its unique, core capability:

- **Council MCP**: Multi-agent deliberation ONLY
- **Code MCP**: File operations (read, write, lint, format, analyze)
- **Git MCP**: Version control operations (add, commit, push, branch management)
- **Cortex MCP**: Memory/RAG operations (query, ingest, cache)
- **Protocol MCP**: Protocol document management
- **Task MCP**: Task lifecycle management
- **Chronicle MCP**: Chronicle entry management
- **ADR MCP**: Architecture decision records
- **Config MCP**: Configuration file management

MCP servers should compose with each other rather than duplicate functionality. For example, a workflow that needs Council deliberation + file write + git commit should call three separate MCP tools in sequence.

**Removed from Council MCP:**
- `council_mechanical_write` → Use `code_write` from Code MCP
- `council_query_memory` → Use `cortex_query` from Cortex MCP
- `council_git_commit` → Use `git_add` + `git_smart_commit` from Git MCP

**Design Pattern:** Composition over duplication

## Consequences

**Positive:**
- Clean architecture with single responsibility per MCP
- No code duplication across MCP servers
- Easier maintenance (changes to file I/O only affect Code MCP)
- Composable workflows enable flexible automation
- Clear boundaries make testing easier
- Follows Unix philosophy: "Do one thing and do it well"

**Negative:**
- Slightly more verbose workflows (multiple tool calls instead of one)
- Requires understanding of which MCP provides which capability
- Potential for increased latency (multiple MCP invocations)
- More complex error handling across MCP boundaries

**Risks:**
- Users may find composition patterns less intuitive initially
- Documentation must clearly explain composition patterns

**Mitigation:**
- Provide clear composition pattern examples in all MCP READMEs
- Document common workflows in MCP Operations Inventory
- Create workflow templates for frequent patterns
