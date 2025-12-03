# Task 022B: MCP User Guides & Architecture Documentation

## Metadata
- **Status**: Done
- **Priority**: Medium
- **Complexity**: Medium
- **Category**: Documentation
- **Estimated Effort**: 4-6 hours
- **Dependencies**: None
- **Parent Task**: 022
- **Created**: 2025-11-28
- **Updated**: 2025-12-01 (Refactored for MCP Architecture)

## Objective

Create quick start guides for connecting to the MCP ecosystem, user tutorials for key workflows (Council, RAG), and update architecture documentation to reflect the 11-server topology.

## Deliverables

1. Create `docs/mcp/QUICKSTART.md` (Connecting Clients to Project Sanctuary)
2. Create `docs/tutorials/01_using_council_mcp.md` (Multi-agent deliberation)
3. Create `docs/tutorials/02_using_cortex_mcp.md` (RAG queries and ingestion)
4. Update `docs/mcp/architecture.md` with detailed component views
5. Create `docs/mcp/diagrams/system_overview_v2.mmd` (Mermaid)
6. Create `docs/INDEX.md` as the entry point for all documentation

## Acceptance Criteria

- [x] `docs/mcp/QUICKSTART.md` enables a new user to configure Claude/Antigravity in < 10 mins
- [x] Tutorial for Council MCP covers dispatch, agent selection, and result interpretation
- [x] Tutorial for Cortex MCP covers querying, ingestion, and scope management
- [x] Architecture documentation accurately reflects all 12 MCP servers and their relationships
- [x] System overview diagram updated to v2 state
- [x] `docs/INDEX.md` provides clear navigation to all resources

## Implementation Steps

### 1. Create Quick Start Guide (1 hour)
- Prerequisites (Python, uv, etc.)
- Configuration steps (`mcp_config.json`)
- Verifying connection
- Troubleshooting common connection issues

### 2. Create Tutorials (2-3 hours)

#### Tutorial 1: The Council
- How to dispatch a task
- Understanding rounds
- Customizing agents (Agent Persona)

#### Tutorial 2: The Cortex
- How to query the knowledge base
- Understanding "Living Memory"
- Ingesting new knowledge

### 3. Update Architecture Documentation (1-2 hours)
- Update `docs/mcp/architecture.md`
- Detail the 12 MCP servers:
  1.  **Admins:** Config, Git, Task
  2.  **Cognitive:** Cortex, Council, Agent Persona, Forge
  3.  **Domain:** Code, Protocol, Chronicle, ADR
- Update diagrams to show data flow between these clusters

### 4. Create Documentation Index (30 minutes)
- Centralize links to all MCP READMEs
- Link to tutorials and guides
- Link to architectural decision records (ADRs)

## Related Protocols

- **Protocol 85**: The Mnemonic Cortex Protocol - Living memory
- **Protocol 89**: The Clean Forge - Quality standards
- **Protocol 115**: The Tactical Mandate - Documentation requirements

## Notes

Refactored to focus on the user experience of the MCP ecosystem. The goal is to make the complex multi-agent system accessible and understandable.
