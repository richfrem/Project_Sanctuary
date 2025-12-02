# Model Context Protocol (MCP) Documentation

**The Nervous System of Project Sanctuary**

## Overview

The Model Context Protocol (MCP) is the architectural backbone of Project Sanctuary, enabling a modular, "Nervous System" design. Instead of a monolithic application, the Sanctuary operates as a constellation of specialized servers that provide tools, resources, and intelligence to the central orchestrator and AI agents.

This architecture allows for:
- **Separation of Concerns:** Each server handles one domain (e.g., Git, Chronicle, RAG).
- **Scalability:** New capabilities can be added as new servers without modifying the core.
- **Interoperability:** Standardized protocol for tools and resources.
- **Security:** Granular control over what each agent can access.

## MCP Server Index

| Server | Domain | Documentation | Status |
|--------|--------|---------------|--------|
| **Cortex** | RAG, Memory, Semantic Search | [README](../../mcp_servers/cognitive/cortex/README.md) | ✅ Active |
| **Chronicle** | Historical Records, Truth | [README](../../mcp_servers/chronicle/README.md) | ✅ Active |
| **Protocol** | Doctrines, Laws | [README](../../mcp_servers/protocol/README.md) | ✅ Active |
| **Council** | Multi-Agent Orchestration | [README](../../mcp_servers/council/README.md) | ✅ Active |
| **Agent Persona** | Agent Roles & Dispatch | [README](../../mcp_servers/agent_persona/README.md) | ✅ Active |
| **Forge** | Fine-Tuning, Model Queries | [README](../../mcp_servers/system/forge/README.md) | ✅ Active |
| **Git Workflow** | Version Control, P101 v3.0 | [README](../../mcp_servers/system/git_workflow/README.md) | ✅ Active |
| **Task** | Task Management | [README](../../mcp_servers/task/README.md) | ✅ Active |
| **Code** | File I/O, Analysis | [README](../../mcp_servers/code/README.md) | ✅ Active |
| **Config** | System Configuration | [README](../../mcp_servers/config/README.md) | ✅ Active |
| **ADR** | Architecture Decisions | [README](../../mcp_servers/document/adr/README.md) | ✅ Active |

## Quick Start

To run an MCP server, use the standard python module syntax from the project root:

```bash
# Example: Run the Cortex MCP Server
python3 -m mcp_servers.cognitive.cortex.server
```

For comprehensive configuration in Claude Desktop or Antigravity, see the [Operations Inventory](mcp_operations_inventory.md).

## Development Standards

- **Testing:** All MCP servers must follow the [Testing Standards](TESTING_STANDARDS.md).
- **Documentation:** Each server must have a README following the standard template.
- **Architecture:** See [MCP Ecosystem Overview](diagrams/mcp_ecosystem_class.mmd).

## Related Resources

- [MCP Operations Inventory](mcp_operations_inventory.md) - Detailed list of all tools
- [RAG Strategies](RAG_STRATEGIES.md) - Deep dive into Cortex architecture
- [Setup Guide](setup_guide.md) - Environment setup
