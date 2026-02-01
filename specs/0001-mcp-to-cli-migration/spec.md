# Feature Specification: MCP-to-CLI Migration

**Feature Branch**: `spec/0001-mcp-to-cli-migration`  
**Category**: Process | Architecture  
**Created**: 2026-01-31  
**Status**: Draft  
**Input**: User description: "Shift architecture from MCP servers to workflows + CLI tools for greater determinism"

## Context

### The Problem with MCP Servers

Project Sanctuary currently relies on MCP (Model Context Protocol) servers running in Podman containers for agent operations. This creates:
1. **Complexity**: Multiple containers to manage, start, and monitor (15+ servers)
2. **Overhead**: Container orchestration, networking, and resource usage
3. **Non-determinism**: MCP server availability, timeouts, and failure modes
4. **Opacity**: Hard to debug what happens inside server calls
5. **Cognitive Load**: Agent must understand MCP tool interfaces instead of following clear workflows

### The SpecKit Antigravity Solution

The imported ADRs and SpecKit toolkit establish a **Spec-Driven Development** approach:

> "Spec-Driven Development flips the script: specifications become executable, directly generating working implementations rather than just guiding them."

**Core Philosophy** (from SpecKit):
- **Intent-driven**: Specifications define the "what" before the "how"
- **Multi-step refinement**: Rather than one-shot code generation
- **Workflow-based execution**: Clear linear progression through phases
- **Deterministic**: Same workflow → same outcome

**SpecKit Workflow Pipeline**:
```
Principles → Specification → Clarification → Plan → Tasks → Implementation
```

| Phase | Command | Purpose |
|-------|---------|---------|
| 0 | `/speckit-constitution` | Establish governing principles |
| 1 | `/speckit-specify` | Define *what* to build |
| 2 | `/speckit-clarify` | De-risk before planning |
| 3 | `/speckit-plan` | Define *how* to build |
| 4 | `/speckit-tasks` | Break into actionable steps |
| 5 | `/speckit-implement` | Execute the tasks |

### Key ADRs

- **ADR-0029**: Hybrid Spec-Driven Development (Factory + Discovery tracks)
- **ADR-0030**: Thick Python / Thin Shim architecture
- **ADR-0031**: Pure Python Orchestration (eliminate shim layer entirely)

### MCP vs SpecKit Approach Comparison

| Aspect | MCP Approach | SpecKit/CLI Approach |
|--------|--------------|---------------------|
| **Execution** | Call MCP tool → server processes → returns result | Call workflow/CLI → direct Python execution |
| **Infrastructure** | Podman containers, networking, health checks | Python scripts, zero containers |
| **Discovery** | MCP tool list from running servers | RLM cache query (`query_cache.py`) |
| **Determinism** | Variable (server availability, timeouts) | High (same input → same output) |
| **Debugging** | Logs scattered across containers | Single process, standard Python debugging |
| **Agent Cognitive Load** | Learn 80+ MCP tool interfaces | Follow 10 workflows, query for specific tools |
| **Startup Time** | 30-60s (container initialization) | <1s (scripts already available) |

### Reference

- **SpecKit Antigravity**: [github.com/richfrem/spec-kit-antigravity](https://github.com/richfrem/spec-kit-antigravity)
- **Agent Debrief**: [docs/antigravity/guides/speckit-debrief.md](../../docs/antigravity/guides/speckit-debrief.md)

## User Scenarios & Testing

### User Story 1 - Agent Tool Discovery (Priority: P1)

As an LLM Agent, I need to discover available tools at runtime so I can perform operations without requiring pre-loaded MCP servers.

**Why this priority**: This is the foundation - agents can't do anything without knowing what tools exist.

**Independent Test**: Query tool cache for "git" and receive matching CLI tools with usage instructions.

**Acceptance Scenarios**:

1. **Given** an initialized RLM cache, **When** agent queries for "chronicle" operations, **Then** it receives CLI commands that replace `chronicle_create_entry` MCP tool
2. **Given** the agent needs to commit code, **When** it queries for git operations, **Then** it receives workflow references for `/workflow-end` and CLI commands for git staging

---

### User Story 2 - Workflow-Based Operations (Priority: P1)

As an LLM Agent, I need to invoke operations via CLI tools and workflows instead of MCP server calls.

**Why this priority**: This is the execution layer that replaces MCP functionality.

**Independent Test**: Execute `python tools/cli.py workflow start --name test --target 001` successfully without any MCP server running.

**Acceptance Scenarios**:

1. **Given** no MCP servers are running, **When** agent starts a workflow, **Then** `WorkflowManager` initializes the spec bundle correctly
2. **Given** an operation previously requiring Chronicle MCP, **When** agent runs equivalent CLI command, **Then** chronicle entry is created correctly

---

### User Story 3 - MCP Function Mapping (Priority: P2)

As a developer, I need a clear mapping from each MCP server operation to its CLI/workflow equivalent so I can migrate incrementally.

**Why this priority**: Enables safe, incremental migration without losing functionality.

**Independent Test**: Reference `docs/operations/mcp/mcp_to_cli_mapping.md` and find CLI equivalent for any MCP operation.

**Acceptance Scenarios**:

1. **Given** the Chronicle MCP operations list (7 operations), **When** I check the mapping, **Then** each has a documented CLI/workflow equivalent
2. **Given** I need to migrate Git MCP, **When** I follow the mapping guide, **Then** I can replace all 8 git operations with CLI commands

---

### User Story 4 - Tool Registration & Inventory (Priority: P2)

As a developer, I need new CLI tools to be automatically registered in the tool inventory so agents can discover them.

**Why this priority**: Ensures the tool discovery system stays current as we add new tools.

**Independent Test**: Create a new tool in `tools/`, run `manage_tool_inventory.py add --path`, verify it appears in `query_cache.py` results.

**Acceptance Scenarios**:

1. **Given** I create a new CLI tool, **When** I run the registration command, **Then** the tool appears in `tool_inventory.json`
2. **Given** a registered tool, **When** I run RLM distillation, **Then** the tool summary appears in `rlm_summary_cache.json`

---

### Edge Cases

- What happens when agent queries for a tool that hasn't been migrated yet?
- How do we handle MCP operations that have no direct CLI equivalent (e.g., streaming responses)?
- What's the fallback if a CLI tool fails?

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide CLI equivalents for all 15 MCP server clusters' operations
- **FR-002**: System MUST support tool discovery via `query_cache.py` for all migrated tools
- **FR-003**: Workflows MUST function without any MCP servers running
- **FR-004**: System MUST maintain backward compatibility during migration (hybrid mode)
- **FR-005**: System MUST provide a migration mapping document for all MCP operations
- **FR-006**: Tool inventory MUST be updated when new CLI tools are created

### Key Entities

- **Tool Inventory** (`tools/tool_inventory.json`): Registry of all CLI tools with metadata
- **RLM Cache** (`.agent/learning/rlm_summary_cache.json`): Semantic summaries for discovery
- **Workflow Manager** (`tools/orchestrator/workflow_manager.py`): Central workflow orchestration
- **CLI Entry Point** (`tools/cli.py`): Unified command interface

## Success Criteria

### Measurable Outcomes

- **SC-001**: All 15 MCP server operations (see `mcp_operations_inventory.md`) have CLI equivalents
- **SC-002**: Agent can complete a full `/workflow-start` → work → `/workflow-end` cycle without MCP servers
- **SC-003**: Tool discovery (`query_cache.py`) returns results for all migrated operations
- **SC-004**: Podman container count for Project Sanctuary reduced from 15+ to 0 (optional)
- **SC-005**: Spec bundle creation works via `WorkflowManager` without MCP

## MCP Servers to Migrate (Reference)

From `mcp_operations_inventory.md`:

1. Chronicle MCP (7 operations)
2. Protocol MCP (5 operations)
3. ADR MCP (5 operations)
4. Task MCP (6 operations)
5. Git MCP (8 operations)
6. RAG Cortex MCP (8 operations)
7. Forge LLM MCP (2 operations)
8. Config MCP (4 operations)
9. Code MCP (11 operations)
10. Agent Persona MCP (5 operations)
11. Council MCP (2 operations)
12. Orchestrator MCP (2 operations)
13. Workflow MCP (2 operations)
14. Learning MCP (Protocol 128 operations)
15. Evolution MCP (Protocol 131 operations)

## Bash Scripts to Migrate (ADR-0031)

Per ADR-0031 (Pure Python Orchestration), these `scripts/bash/` files need conversion:

| Script | Size | Priority | Target |
|--------|------|----------|--------|
| `update-agent-context.sh` | 26KB | High | `tools/cli.py context update` |
| `create-new-feature.sh` | 10KB | High | Merge into `WorkflowManager` |
| `check-prerequisites.sh` | 5KB | Medium | `tools/cli.py prereqs check` |
| `common.sh` | 5KB | Medium | `tools/utils/shell_compat.py` |
| `setup-plan.sh` | 2KB | Low | Part of `/speckit-plan` |
| `workflow-start.sh` | 655B | ✅ Thin shim | Keep as pass-through |
| `workflow-retrospective.sh` | 334B | ✅ Thin shim | Keep as pass-through |
| `workflow-end.sh` | 314B | ✅ Thin shim | Keep as pass-through |

**Goal**: Eliminate complex bash logic; keep only thin shims that `exec python3`.

## Related ADRs

- [ADR-035: Hybrid Spec-Driven Development Workflow](../../ADRs/035_hybrid_spec_driven_development_workflow.md)
- [ADR-036: Workflow Shim Architecture](../../ADRs/036_workflow_shim_architecture.md)
- [ADR-096: Pure Python Orchestration](../../ADRs/096_pure_python_orchestration.md)

