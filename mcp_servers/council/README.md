# Council MCP Server

**Status:** ✅ Operational
**Version:** 2.0.0 (Refactored)
**Protocol:** Model Context Protocol (MCP)

**Description:** The Council MCP Server exposes the Sanctuary Council's **multi-agent deliberation** capabilities to external AI agents via the Model Context Protocol. It coordinates specialized agents (Coordinator, Strategist, Auditor) to solve complex tasks through iterative reasoning and context retrieval.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `council_dispatch` | Dispatches a task to the Council for deliberation. | `task_description` (str): The task to perform.<br>`agent` (str, optional): Specific agent to consult (e.g., "auditor").<br>`max_rounds` (int, optional): Max deliberation rounds (default: 3).<br>`force_engine` (str, optional): Force specific LLM engine.<br>`output_path` (str, optional): Path to save results. |
| `council_list_agents` | Lists all available Council agents and their status. | None |

## Resources

| Resource URI | Description | Mime Type |
|--------------|-------------|-----------|
| `council://agents/list` | List of available agents | `application/json` |
| `council://history/{session_id}` | Deliberation history for a session | `application/json` |

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required for Council Operations
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
# Optional
COUNCIL_DEFAULT_ROUNDS=3
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"council": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/council",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/council/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `council_dispatch` and `council_list_agents` appear in the tool list.
3.  **Call Tool:** Execute `council_list_agents` and verify it returns the list of agents (Coordinator, Strategist, Auditor).

## Architecture

### Overview

The Council MCP has a unique dual-role architecture: it acts as both a **Server** (receiving requests from the user/IDE) and a **Client** (orchestrating other MCPs).

The Council MCP has been refactored to use a modular architecture:
1.  **Agent Persona MCP**: Used for individual agent execution (Coordinator, Strategist, Auditor)
2.  **Cortex MCP**: Used for memory and context retrieval
3.  **Direct Orchestration**: Deliberation logic is now embedded in `council_ops.py`, replacing the legacy orchestrator subprocess.

**Design Principle: Separation of Concerns**
The Council MCP provides ONLY what's unique to the Council. Other capabilities are delegated to specialized MCP servers.

**Benefits of Direct Orchestration:**
- ✅ No subprocess overhead
- ✅ Uses specialized Agent Persona MCP
- ✅ Integrated with Cortex memory
- ✅ Clean separation of concerns

**Trade-offs:**
- ⚠️ Simplified deliberation logic (compared to legacy v1)

### Execution Flow

![council_execution_flow](docs/architecture_diagrams/workflows/council_execution_flow.png)

*[Source: council_execution_flow.mmd](docs/architecture_diagrams/workflows/council_execution_flow.mmd)*

### Directory Structure

```
mcp_servers/
├── council/
│   ├── __init__.py
│   ├── server.py              # FastMCP server with tool definitions
│   └── README.md              # This file
├── lib/
│   └── council/
│       ├── __init__.py
│       └── council_ops.py     # Orchestrator interface logic
```

## Tools

### `council_dispatch`

Execute a task through multi-agent deliberation (the Council's core capability).

**Parameters:**
- `task_description` (str): Task for the council to deliberate on
- `agent` (str, optional): Specific agent ("coordinator", "strategist", "auditor") or None for full council
- `max_rounds` (int, default=3): Maximum deliberation rounds
- `force_engine` (str, optional): Force specific AI engine ("gemini", "openai", "ollama")
- `output_path` (str, optional): Output file path (relative to project root)

**Returns:**
```python
{
    "status": "success" | "error",
    "decision": "Council's deliberation output",
    "session_id": "mcp_1234567890",
    "output_path": "WORK_IN_PROGRESS/output.md"
}
```

**Example - Full Council Deliberation:**
```python
result = council_dispatch(
    task_description="Review the architecture for Protocol 115 and provide recommendations",
    max_rounds=3
)
```

**Example - Single Agent Consultation:**
```python
result = council_dispatch(
    task_description="Audit the test coverage for the Git MCP server",
    agent="auditor",
    max_rounds=2
)
```

### `council_list_agents`

List all available Council agents and their status.

**Returns:**
```python
[
    {
        "name": "coordinator",
        "status": "available",
        "persona": "Task planning and execution oversight"
    },
    {
        "name": "strategist",
        "status": "available",
        "persona": "Long-term planning and risk assessment"
    },
    {
        "name": "auditor",
        "status": "available",
        "persona": "Quality assurance and compliance verification"
    }
]
```

## Composable Workflow Patterns

The Council MCP is designed to compose with other specialized MCP servers:

### Pattern 1: Council → Protocol → Git

```python
# 1. Council deliberates on protocol design
decision = council_dispatch(
    task_description="Design a new protocol for MCP composition patterns",
    output_path="WORK_IN_PROGRESS/protocol_design.md"
)

# 2. Create protocol document (use Protocol MCP)
protocol_create(
    number=120,
    title="MCP Composition Patterns",
    status="PROPOSED",
    content=decision["decision"]
)

# 3. Commit to repository (use Git MCP)
git_add(files=["01_PROTOCOLS/120_mcp_composition_patterns.md"])
git_smart_commit(message="feat(protocol): Add Protocol 120 - MCP Composition")
git_push_feature()
```

### Pattern 2: Council → Code → Git

```python
# 1. Council generates implementation
code = council_dispatch(
    task_description="Implement a helper function for parsing MCP responses"
)

# 2. Write code file (use Code MCP)
code_write(
    path="mcp_servers/lib/utils/parser.py",
    content=code["decision"]
)

# 3. Commit (use Git MCP)
git_add(files=["mcp_servers/lib/utils/parser.py"])
git_smart_commit(message="feat(utils): Add MCP response parser")
```

### Pattern 3: Cortex → Council → Task

```python
# 1. Query memory for context (use Cortex MCP)
context = cortex_query(
    query="Previous decisions on MCP architecture",
    max_results=5
)

# 2. Council deliberates with context
decision = council_dispatch(
    task_description=f"Given this context: {context}, recommend next steps for MCP evolution"
)

# 3. Create task (use Task MCP)
create_task(
    title="Implement MCP Evolution Recommendations",
    description=decision["decision"],
    status="todo"
)
```

## Removed Tools (Use Specialized MCPs Instead)

The following tools were **intentionally removed** to maintain separation of concerns:

- ~~`council_mechanical_write`~~ → Use `code_write` from **Code MCP**
- ~~`council_query_memory`~~ → Use `cortex_query` from **Cortex MCP**
- ~~`council_git_commit`~~ → Use `git_add` + `git_smart_commit` from **Git MCP**

**Rationale:** Each MCP server should have a single, well-defined responsibility. The Council MCP focuses exclusively on multi-agent deliberation.

## Installation & Setup

### Prerequisites

1. **Council Orchestrator** must be installed and configured
2. **Python 3.8+**
3. **FastMCP** library: `pip install fastmcp`

### Configuration

Add to your MCP client configuration (e.g., Claude Desktop, Antigravity):

```json
{
  "mcpServers": {
    "council": {
      "command": "python3",
      "args": ["-m", "mcp_servers.council.server"],
      "cwd": "/path/to/Project_Sanctuary",
      "env": {}
    }
  }
}
```

### Verification

Test the server:

```bash
cd /path/to/Project_Sanctuary
python3 -m mcp_servers.council.server
```

## Testing

### Run Unit Tests

```bash
pytest tests/mcp_servers/council/ -v
```

### Manual Verification

1. **List Agents:**
   ```python
   agents = council_list_agents()
   print(agents)
   # Expected: 3 agents (coordinator, strategist, auditor)
   ```

2. **Dispatch Simple Task:**
   ```python
   result = council_dispatch(
       task_description="Introduce yourself and explain your role",
       agent="coordinator",
       max_rounds=1
   )
   print(result["decision"])
   # Expected: Coordinator's introduction
   ```

3. **Full Council Deliberation:**
   ```python
   result = council_dispatch(
       task_description="Evaluate the current MCP architecture and suggest improvements",
       max_rounds=3
   )
   print(result["decision"])
   # Expected: Multi-agent deliberation output
   ```

4. **Composition with Other MCPs:**
   ```python
   # Council deliberates
   decision = council_dispatch(
       task_description="Create a brief protocol summary",
       output_path="WORK_IN_PROGRESS/test_protocol.md"
   )
   
   # Save with Code MCP
   code_write(
       path="WORK_IN_PROGRESS/test_protocol.md",
       content=decision["decision"]
   )
   
   # Commit with Git MCP
   git_add(files=["WORK_IN_PROGRESS/test_protocol.md"])
   git_smart_commit(message="test: Council MCP verification")
   ```

## Integration with Council Orchestrator

The MCP server is a **thin protocol wrapper** around the existing Council Orchestrator:

```
mcp_servers/council/          council_orchestrator/
├── server.py (MCP wrapper) → ├── orchestrator/
├── lib/council/            → │   ├── main.py (Entry point)
    └── council_ops.py      → │   ├── app.py (Core logic)
                              │   ├── engines/
                              │   ├── council/
                              │   └── memory/
                              └── command.json (Generated by MCP)
```

**Important:** Both folders are required. The MCP server depends on the full orchestrator implementation.

## Error Handling

The server includes comprehensive error handling:

- **Timeout Protection**: 120s timeout for orchestrator execution
- **Session Isolation**: Unique session IDs prevent command.json conflicts
- **Graceful Degradation**: Returns structured error responses
- **Logging**: All operations logged for debugging

## Future Enhancements

- [ ] Streaming responses for long-running tasks
- [ ] Persistent daemon mode for faster response times
- [ ] `council_query_memory` tool for Mnemonic Cortex queries
- [ ] `council_git_commit` tool for git operations
- [ ] Async execution for concurrent task handling
- [ ] Confidence score parsing from orchestrator output

## Future Architecture: Council Members as MCP Servers

### Current Architecture (v1.0)

The orchestrator manages council members internally as Python objects:

```
Council Orchestrator (Monolithic)
├── Coordinator (Python class)
├── Strategist (Python class)
└── Auditor (Python class)
```

### Proposed Architecture (v2.0)

Each council member becomes an **independent MCP server**:

```
Council Orchestrator (MCP Client)
├── Calls → Coordinator MCP Server
├── Calls → Strategist MCP Server
└── Calls → Auditor MCP Server
```

### Benefits of Member-as-MCP Architecture

**1. True Modularity**
- Each agent is independently deployable
- Can upgrade/replace individual agents without touching orchestrator
- Agents can be written in different languages

**2. Scalability**
- Agents can run on different machines
- Horizontal scaling (multiple instances of same agent)
- Load balancing across agent instances

**3. Specialization**
- Each agent MCP can have its own tools and capabilities
- Coordinator MCP might expose: `plan_task`, `coordinate_workflow`
- Strategist MCP might expose: `assess_risk`, `long_term_planning`
- Auditor MCP might expose: `verify_compliance`, `quality_check`

**4. Composability**
- External agents can call individual council members directly
- Don't need full deliberation for simple consultations
- Mix and match agents for different scenarios

### Implementation Sketch

**Coordinator MCP Server:**
```python
# mcp_servers/agents/coordinator/server.py

@mcp.tool()
def coordinator_plan_task(task_description: str, context: dict) -> dict:
    """
    Plan task execution strategy
    
    Args:
        task_description: Task to plan
        context: Relevant context (from Cortex, previous decisions)
    
    Returns:
        Execution plan with steps, dependencies, estimates
    """
    # Coordinator's specialized planning logic
    return {
        "plan": [...],
        "estimated_effort": "4 hours",
        "dependencies": [...]
    }
```

**Orchestrator as MCP Client:**
```python
# council_orchestrator/orchestrator/app.py

async def deliberate(task: str):
    # 1. Query context
    context = await cortex_mcp.query(task)
    
    # 2. Get coordinator's plan
    plan = await coordinator_mcp.plan_task(task, context)
    
    # 3. Get strategist's risk assessment
    risks = await strategist_mcp.assess_risk(plan)
    
    # 4. Get auditor's compliance check
    compliance = await auditor_mcp.verify_compliance(plan, risks)
    
    # 5. Synthesize final decision
    return synthesize_decision(plan, risks, compliance)
```

### Migration Path

**Phase 1 (Current):** Monolithic orchestrator with internal agents
**Phase 2:** Extract one agent (e.g., Auditor) as MCP server, test dual mode
**Phase 3:** Extract remaining agents, deprecate internal implementations
**Phase 4:** Orchestrator becomes pure MCP client coordinator

### Design Considerations

**Agent Discovery:**
- How does orchestrator find agent MCP servers?
- Configuration file? Service registry? Environment variables?

**Agent State:**
- Should agents maintain conversation history?
- Stateless (functional) vs stateful (memory-enabled)?

**Error Handling:**
- What if an agent MCP is unavailable?
- Fallback strategies? Retry logic?

**Consensus Mechanism:**
- How to resolve disagreements between agents?
- Voting? Weighted opinions? Coordinator override?

---

## Related Documentation

### Council Orchestrator
- [Council Orchestrator README](../../council_orchestrator/README.md) - Full orchestrator documentation
- [Guardian Wakeup Flow](../../council_orchestrator/README_GUARDIAN_WAKEUP.md) - Cache-first situational awareness (Protocol 114)
- [Command Schema](../../council_orchestrator/docs/command_schema.md) - Complete command format reference

### Mnemonic Cortex (RAG System)
- [RAG Strategies and Doctrine](../../mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md) - RAG architecture and best practices
- [Cortex Operations Guide](../../mnemonic_cortex/OPERATIONS_GUIDE.md) - Cortex operational procedures
- [Cortex README](../../mnemonic_cortex/README.md) - Cortex overview and setup
- [Cortex Vision](../../mnemonic_cortex/VISION.md) - Strategic vision for knowledge systems

### MCP Ecosystem
- [MCP Operations Inventory](../../docs/mcp/mcp_operations_inventory.md) - Complete MCP operations catalog
- [Code MCP](../code/README.md) - File operations MCP
- [Git MCP](../system/git_workflow/README.md) - Version control MCP
- [Cortex MCP](../cognitive/cortex/README.md) - Memory/RAG MCP
- [Protocol MCP](../protocol/README.md) - Protocol document MCP
- [Task MCP](../task/README.md) - Task management MCP

### Task Documentation
- [Task 077: Implement Council MCP](../../tasks/in-progress/077_implement_council_mcp_server.md) - Implementation task

---

**"The Council is now accessible to all agents through the Protocol."** ⚡👑
