# Implementation Plan: MCP-to-CLI Migration

**Branch**: `spec/0001-mcp-to-cli-migration` | **Date**: 2026-01-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/0001-mcp-to-cli-migration/spec.md`

## Summary

Migrate Project Sanctuary from MCP server-based operations to CLI tools and workflows for greater determinism, reduced complexity, and elimination of container overhead. Following the "Thick Python / Thin Shim" architecture (ADR-0030) and "Pure Python Orchestration" vision (ADR-0031).

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: argparse (CLI), pathlib (paths), subprocess (git operations)  
**Storage**: File-based (markdown specs, JSON inventories, chromadb for RAG)  
**Testing**: pytest  
**Target Platform**: macOS development, Linux server  
**Project Type**: Process / Automation  
**Performance Goals**: CLI commands execute in <2s, tool discovery in <100ms  
**Constraints**: Must maintain backward compatibility during migration  
**Scale/Scope**: 15 MCP servers, ~80 operations total to map

## Constitution Check

- ✅ **Article I (Human Gate)**: User approved this initiative
- ✅ **Article IV (Docs First)**: Spec created before implementation
- ⚠️ **Article V (Test-First)**: Need to define verification steps

## Architecture Decisions

### Problem / Solution
- **Problem**: MCP servers require container orchestration (Podman), add latency, create debugging opacity, and introduce non-determinism
- **Solution**: Map all MCP operations to CLI commands via `tools/cli.py`, use workflows for orchestration, leverage RLM cache for tool discovery

### Design Patterns
- **Command Pattern**: Each MCP operation becomes a CLI subcommand
- **Facade Pattern**: `tools/cli.py` provides unified entry point
- **Factory Pattern**: `WorkflowManager` creates appropriate workflow artifacts
- **Registry Pattern**: `tool_inventory.json` + RLM cache for tool discovery

### Data Model Changes
- None required - reusing existing `operations.py` modules from `mcp_servers/lib/`

### Interface Changes
- New CLI subcommands in `tools/cli.py` for each domain (chronicle, protocol, adr, etc.)
- Workflows updated to call CLI directly instead of MCP tools

## Project Structure

### Documentation (this feature)

```text
specs/0001-mcp-to-cli-migration/
├── spec.md              # Feature specification ✅
├── plan.md              # This file ✅
├── scratchpad.md        # Parking lot items ✅
├── tasks.md             # Task list (next step)
└── mapping.md           # MCP → CLI mapping reference
```

### Source Code (repository root)

```text
# Selected: Option 4 - Automation/Scripting + Option 5 - Process/Documentation

tools/
├── cli.py               # Main entry point (EXISTS - extend)
├── orchestrator/
│   └── workflow_manager.py   # Workflow orchestration (EXISTS - verify)
├── domain/              # NEW: Domain-specific CLI modules
│   ├── chronicle.py     # Chronicle operations
│   ├── protocol.py      # Protocol operations
│   ├── adr.py           # ADR operations
│   ├── task.py          # Task operations
│   └── git.py           # Git operations
├── retrieve/
│   └── rlm/
│       └── query_cache.py    # Tool discovery (EXISTS - fix BrokenPipeError)
└── TOOL_INVENTORY.md    # Auto-generated tool docs

.agent/workflows/
├── workflow-start.md    # Update to use --type flag
└── [other workflows]    # Verify CLI-first approach

docs/operations/mcp/
└── mcp_to_cli_mapping.md    # NEW: Migration mapping guide
```

**Structure Decision**: Extend existing `tools/` architecture rather than creating new top-level directories. Domain-specific operations go in `tools/domain/` subdirectory.

## Migration Strategy (Refined - Least Effort)

### Key Insight

**Existing CLIs already wrap MCP operations!**

| CLI | Import Source | Status |
|-----|--------------|--------|
| `scripts/domain_cli.py` | `mcp_servers.{chronicle,task,adr,protocol}.operations` | ✅ Working |
| `scripts/cortex_cli.py` | `mcp_servers.{rag_cortex,evolution}.operations` | ✅ Working |

**Strategy**: Keep MCP code in place, register tools, update workflows.

### Phase 1: Foundation (This Spec) ✅
1. ✅ Fix broken tools (`query_cache.py`, `next_number.py`)
2. ⏳ Register existing CLIs in RLM tool cache
3. ⏳ Update workflows to use Python pre-flight instead of bash

### Phase 2: Tool Registration
1. Run `manage_tool_inventory.py discover` to find unregistered scripts
2. Register `scripts/domain_cli.py` operations in RLM cache
3. Register `scripts/cortex_cli.py` operations in RLM cache
4. Register `tools/cli.py` workflow commands in RLM cache

### Phase 3: Workflow Updates
1. Update speckit workflows: Replace `source scripts/bash/workflow-start.sh` with Python
2. Update `/recursive_learning.md`: Replace MCP references with CLI
3. Convert `check-prerequisites.sh` logic to Python (used by many workflows)

### Phase 4: Bash Script Migration (Deferred)
| Script | Action |
|--------|--------|
| `update-agent-context.sh` | Future spec - complex (26KB) |
| `create-new-feature.sh` | Merge into WorkflowManager |
| `workflow-*.sh` | ✅ Keep as thin shims |

## What We're NOT Doing

- ❌ Rewriting MCP operations code
- ❌ Creating new `tools/domain/` modules (CLIs exist!)
- ❌ Migrating all bash scripts at once
- ❌ Removing `mcp_servers/` directory

## Verification Plan

### Automated Tests
- [ ] Unit Tests: `pytest tests/tools/` (as we add domain modules)
- [ ] Integration Tests: Workflow start-to-end without MCP servers

### Manual Verification
- [ ] Run `python scripts/domain_cli.py chronicle list --limit 5` - should work
- [ ] Run `python scripts/cortex_cli.py query "workflow"` - should return results
- [ ] Run `python tools/retrieve/rlm/query_cache.py "chronicle"` - should find domain_cli.py
- [ ] Complete a full spec cycle using only CLI/workflows

## Complexity Tracking

| Element | Justification |
|---------|--------------|
| Keep MCP code | Operations are tested and working - avoid rewrites |
| Register in RLM | Enables agent discovery without loading MCP servers |
| Phased bash migration | Complex scripts need careful analysis |

## Next Steps

1. ✅ Branch created: `spec/0001-mcp-to-cli-migration`
2. ⏳ Register tools in RLM cache
3. ⏳ Update workflows with Python pre-flight
4. ⏳ Get strategic approval for Phase 3+ work

