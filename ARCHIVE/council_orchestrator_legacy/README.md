# Council Orchestrator - Refactoring in Progress

**Status:** Partial migration to MCP architecture  
**Date:** 2025-11-30

## Current State

This directory contains the **core orchestration logic** that is being refactored to use the new MCP architecture.

### Active Files (Pending Refactoring)

Only one file remains in `council_orchestrator/orchestrator/`:

1. **`app.py`** (96KB) - Core multi-agent deliberation logic
   - **Target:** Move to `mcp_servers/lib/council/orchestrator.py`
   - **Refactor:** Replace internal agent/engine/memory calls with MCP tool calls
   - **Use:** Agent Persona MCP for agents, Cortex MCP for memory

### Already Migrated to Council MCP

- **`packets/`** → `mcp_servers/lib/council/packets/` ✅
  - Round packet system for tracking deliberation
  - Files: `schema.py`, `emitter.py`, `aggregator.py`, `__init__.py`

### Archived Components

The following components have been **archived** to `ARCHIVE/council_orchestrator/orchestrator/`:

#### Already Migrated to MCPs
- **`council/`** → Agent Persona MCP (`mcp_servers/agent_persona/`)
- **`engines/`** → Agent Persona MCP (`mcp_servers/lib/agent_persona/llm_client.py`)
- **`memory/`** → Cortex MCP (`mcp_servers.rag_cortex/`)
- **`handlers/cache_wakeup_handler.py`** → Cortex MCP (`cortex_guardian_wakeup` tool)
- **`gitops.py`** → Git MCP (`mcp_servers/system/git_workflow/`)
- **`substrate_monitor.py`** → Agent Persona MCP (engine selection logic)

#### Obsolete/Specialized
- **`main.py`** - Entry point (replaced by Council MCP server)
- **`sentry.py`** - File monitoring (replaced by MCP tool calls)
- **`commands.py`** - Command validation (replaced by MCP tool validation)
- **`regulator.py`** - Token flow regulation (not needed with new architecture)
- **`optical.py`** - Optical compression (specialized, archived)
- **`events.py`** - Event logging (use standard logging)
- **`executor.py`** - Execution logic (unclear purpose, archived)
- **`adaptation_packets.py`** - Adaptation packet generation (Cortex MCP handles this)
- **`config/`** - Configuration files (obsolete)

## Next Steps (Task 60268594)

1. **Refactor `app.py`:**
   - Replace internal agent creation with Agent Persona MCP calls
   - Use `persona_dispatch` for each agent's turn
   - Maintain deliberation loop logic

2. **Move to `mcp_servers/lib/council/`:**
   - Create `orchestrator.py` (refactored `app.py`)
   - Create `commands.py` (validation logic)
   - Create `packets/` (round packet system)

3. **Update Council MCP Server:**
   - Update `mcp_servers/council/server.py` to use new orchestrator
   - Remove dependency on `council_orchestrator/` directory

4. **Archive Remaining Files:**
   - Once refactoring is complete, move `council_orchestrator/` to `ARCHIVE/`

## Related Documentation

- **Agent Persona MCP:** `mcp_servers/agent_persona/README.md`
- **Council MCP:** `mcp_servers/council/README.md`
- **Task 079:** Refactor Agent Persona MCP Terminology (✅ Complete)
- **Task 080:** Migrate and Archive Legacy Council Orchestrator (✅ Complete)
- **Task 60268594:** Refactor Council Orchestrator (🔄 Next)
