# Task 077: Implement Council MCP Server (Agent Orchestrator)

**Status:** done
**Priority:** High
**Lead:** Antigravity
**Dependencies:** Task 072 (Code MCP), Task 055 (Git MCP)
**Related Documents:** council_orchestrator/, mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md, mnemonic_cortex/OPERATIONS_GUIDE.md

---

## 1. Objective

Implement the **Council MCP Server**, which serves as the interface for the **Agent Orchestrator**. This MCP server will expose tools to manage, query, and coordinate the Council of Agents (Architect, Guardian, etc.). It acts as the bridge between the high-level orchestration logic and the MCP ecosystem.

## 2. Deliverables

1.  **Council MCP Server Implementation** (`mcp_servers/council/`)
    *   `server.py`: Main FastMCP server instance
    *   `tools/`: Tool definitions
2.  **Core Tools**
    *   `council_list_agents`: List available agents and their status
    *   `council_dispatch`: Dispatch a task to a specific agent
    *   `council_broadcast`: Broadcast a message to all agents
    *   `council_get_consensus`: Retrieve consensus on a topic (if applicable)
3.  **Integration**
    *   Connect to `council_orchestrator` package
    *   Ensure access to `mnemonic_cortex` for context
4.  **Testing**
    *   Unit tests for tools
    *   Integration tests with Orchestrator
5.  **Documentation**
    *   README.md with usage and examples

## 3. Acceptance Criteria

- [x] Server runs and connects to Claude Desktop / Antigravity
- [x] Can list all active agents
- [x] Can dispatch a simple task to an agent and get a response
- [x] Error handling for unavailable agents or invalid tasks
- [x] Comprehensive test suite passing

## 4. Implementation Plan

### Phase 1: Analysis & Design
- Review `council_orchestrator` architecture
- Define tool schema

### Phase 2: Server Scaffold
- Create directory structure
- Implement basic `server.py`

### Phase 3: Tool Implementation
- Implement `list_agents`
- Implement `dispatch`
- Implement `broadcast`

### Phase 4: Verification
- Run unit tests
- Manual verification via MCP client
