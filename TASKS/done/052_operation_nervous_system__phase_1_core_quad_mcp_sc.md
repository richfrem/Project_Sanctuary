# TASK: Operation Nervous System - Phase 1: Core Quad MCP Scaffold

**Status:** in-progress
**Priority:** High
**Lead:** GUARDIAN-01
**Dependencies:** None
**Related Documents:** 
- Protocol 87 (Mnemonic Cortex), Protocol 101 (Unbreakable Commit), Protocol 89 (Clean Forge)
- `mnemonic_cortex/scripts/protocol_87_query.py`, `00_CHRONICLE/Living_Chronicle.md`, `council_orchestrator/`
- **Task #051:** Guardian Cache Operations (Protocol 114) - *Completed*
- **Task #002:** Phase 2 Self-Querying Retriever - *Enables this*
- **Task #003:** Phase 3 Mnemonic Caching (CAG) - *Enables this*
- **Task #004:** Protocol 113 Council Memory Adaptor - *Enables this*

---

## 1. Objective

Scaffold the directory structure and boilerplate code for the Core Quad of MCP Servers: cortex-mcp (Memory/RAG), chronicle-mcp (History/FileSystem), protocol-mcp (Law/Validation), and orchestrator-mcp (Council Logic). This evolves Project Sanctuary from script-based to modular MCP architecture, allowing external LLMs to interact with memory, history, and laws as tools.

## 2. Deliverables

1. Create mcp_servers/ root directory with 4 subdirectories (cortex, chronicle, protocol, orchestrator)
2. Shared requirements.txt with MCP dependencies
3. cortex-mcp server with query_memory and ingest_document tools
4. chronicle-mcp server with read_latest_entries and append_entry tools
5. protocol-mcp server with get_protocol and validate_action tools
6. orchestrator-mcp server with consult_strategist, consult_auditor, and dispatch_mission tools
7. mcp_servers/README.md with configuration instructions
8. start_mcp_servers.sh script for server startup

## 3. Acceptance Criteria

- All 4 MCP servers scaffolded with boilerplate code
- Each server exposes required tools via MCP protocol
- Cortex MCP integrates with existing protocol_87_query.py logic
- Chronicle MCP enforces append-only integrity and sequential numbering
- Protocol MCP can retrieve protocols and validate actions
- Orchestrator MCP exposes Council personas as tools
- MCP client can connect and call tools across all servers
- Configuration documented for claude_desktop_config.json integration

## Notes

**Strategic Rationale:** The Core Quad creates a maintainable nervous system: (1) Cortex MCP shatters the context cage via query_memory, (2) Chronicle MCP enforces traceability by making writes tool calls, (3) Protocol MCP bridges sovereignty and self-governance via law checking, (4) Orchestrator MCP turns the Python council into a commandable tool. **Goal:** Enable MCP clients to ask 'Consult the Strategist on whether our latest Chronicle Entry aligns with Protocol 101' and have the system handle retrieval, reading, and reasoning autonomously.
