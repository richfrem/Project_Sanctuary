# Archived Council Orchestrator Assets

This directory contains archived assets from the legacy `council_orchestrator` folder.

**Archived Date:** 2025-11-30

## Contents

- `docs/`: Legacy documentation (moved from `council_orchestrator/docs/`)
- `schemas/`: Legacy JSON schemas
- `scripts/`: Legacy utility scripts
- `tests/`: Legacy test suite
- `logs/`: Archived logs
- `runtime/`: Archived runtime state
- `WORK_IN_PROGRESS/`: Archived WIP files
- `command_results/`: Archived command results

## Active Code

The active code for the Council Orchestrator has been split:

1.  **Agent Logic:** Migrated to `mcp_servers/agent_persona/` (Agent Persona MCP).
2.  **Orchestration Logic:** Currently resides in `council_orchestrator/orchestrator/` (pending refactoring to `mcp_servers/council/`).
3.  **Documentation:** Migrated to `docs/legacy/council_orchestrator/`.

## Migration Status

- **Agent Persona MCP:** Fully functional and decoupled.
- **Council MCP:** Currently wraps the legacy orchestrator.
- **Legacy Orchestrator:** Being refactored to use Agent Persona MCP, then will be fully migrated to Council MCP.
