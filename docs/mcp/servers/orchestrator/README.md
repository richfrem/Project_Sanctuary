# Orchestrator MCP Server Documentation

## Overview

Orchestrator MCP is a **general-purpose orchestrator** that coordinates strategic missions across ALL MCPs. It manages multi-phase workflows, task lifecycle, and cross-domain coordination.

## Key Concepts

- **Strategic Coordination:** Manages high-level missions spanning multiple MCPs
- **Multi-Phase Workflows:** Research → Design → Implement → Verify → Document
- **Task Lifecycle:** Creates, tracks, and completes tasks across the ecosystem
- **Cross-Domain:** Delegates to specialized MCPs (Council, Task, Chronicle, Protocol, etc.)

## Architecture

```
External LLM → Orchestrator MCP (Server)
                    ↓ (Client to many MCPs)
            Council MCP, Task MCP, Chronicle MCP,
            Protocol MCP, Code MCP, Git MCP, Cortex MCP
```

## Server Implementation

- **Server Code:** [mcp_servers/orchestrator/server.py](../../../mcp_servers/orchestrator/server.py)
- **Operations:** [mcp_servers/orchestrator/operations.py](../../../mcp_servers/orchestrator/operations.py)

## Testing

- **Test Suite:** [tests/mcp_servers/orchestrator/](../../../tests/mcp_servers/orchestrator/)
- **Status:** 🔄 In Progress

## Operations

### `orchestrator_dispatch_mission`
Dispatch a high-level mission to coordinate across multiple MCPs

**Example:**
```python
orchestrator_dispatch_mission(
    mission="Implement Protocol 120 - MCP Composition Patterns",
    phases=["research", "design", "implement", "verify", "document"]
)
```

### `orchestrator_run_strategic_cycle`
Execute a full Strategic Crucible Loop

### `get_orchestrator_status`
Get current status of the orchestrator

### `list_recent_tasks`
List recent tasks managed by orchestrator

### `get_task_result`
Get result of a specific task

### `create_cognitive_task`
Create a cognitive task for Council deliberation

### `create_development_cycle`
Create a staged development cycle task

## Relationship to Council MCP

Orchestrator MCP **delegates deliberation tasks** to Council MCP when multi-agent discussion is needed as part of a larger strategic workflow.

**Example Workflow:**
```
Orchestrator: "Implement Protocol 120"
  ↓
  Phase 1: Research → Calls Council MCP for strategic analysis
    ↓
    Council MCP → Calls Agent Persona MCP (agents deliberate)
  ↓
  Phase 2: Design → Calls Protocol MCP to create protocol
  ↓
  Phase 3: Implement → Calls Code MCP, Git MCP
  ↓
  Phase 4: Verify → Calls Council MCP for review
  ↓
  Phase 5: Document → Calls Chronicle MCP
```

## Performance

- **Latency:** Minutes to hours (multi-phase workflows)
- **Bottleneck:** Depends on phase (often Council MCP deliberations)
- **Scalability:** Coordinates long-running, complex workflows

## Related Documentation

- **[Council vs Orchestrator](../council/council_vs_orchestrator.md)** - Relationship explanation

## Status

🔄 **In Development** - Core operations defined, testing in progress
