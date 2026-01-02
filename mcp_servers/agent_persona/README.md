# Agent Persona MCP Server

**Description:** The Agent Persona MCP provides access to specialized AI agent personas that can be used for domain-specific tasks. Each persona has a unique role, expertise, and reasoning style.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `persona_dispatch` | Execute a task using a specific persona agent. | `role` (str): Persona role (e.g., "auditor").<br>`task` (str): Task to execute.<br>`context` (dict, optional): Context data.<br>`maintain_state` (bool): Persist history.<br>`engine` (str, optional): AI engine.<br>`model_name` (str, optional): Specific model. |
| `persona_list_roles` | List all available persona roles (built-in and custom). | None |
| `persona_get_state` | Get conversation state for a specific persona role. | `role` (str): Persona role. |
| `persona_reset_state` | Reset conversation state for a specific persona role. | `role` (str): Persona role. |
| `persona_create_custom` | Create a new custom persona definition. | `role` (str): Unique identifier.<br>`persona_definition` (str): Full instruction text.<br>`description` (str): Brief description. |

## Resources

*No resources currently exposed.*

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required for AI Engines
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
# Optional
PERSONA_STORAGE_DIR=mcp_servers/agent_persona/personas
STATE_STORAGE_DIR=mcp_servers/agent_persona/state
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"agent_persona": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/agent_persona",
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
pytest mcp_servers/agent_persona/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `persona_dispatch` and `persona_list_roles` appear in the tool list.
3.  **Call Tool:** Execute `persona_list_roles` and verify it returns the built-in roles (Coordinator, Strategist, Auditor).

## Architecture

### Design Pattern: Configurable Service
The Agent Persona MCP follows the **"Configurable Service"** pattern - similar to Code MCP or Config MCP, but for AI personas instead of files.

**Key Components:**
1.  **Persona Files** (`personas/*.txt`) - Define agent behavior and expertise
2.  **State Management** (`state/*.json`) - Maintain conversation history
3.  **Engine Integration** - Uses council_orchestrator's engine selection
4.  **PersonaAgent Class** - Reuses proven agent implementation

### Integration with Council Orchestrator
The Council MCP delegates agent execution to this server.

```
Agent Persona MCP (Lightweight)
├── Persona Definitions (*.txt files)
├── State Management (conversation history)
└── Engine Selection (delegates to council_orchestrator)
    ├── select_engine(config) → Healthy engine instance
    └── PersonaAgent(engine, persona_file, state_file)
```

List all available persona roles.

**Returns:**
```python
{
    "built_in": ["coordinator", "strategist", "auditor"],
    "custom": ["security_reviewer", "performance_analyst"],
    "total": 5,
    "persona_dir": "/path/to/personas"
}
```

### 3. `persona_get_state`

Get conversation state for a specific role.

**Example:**
```python
state = persona_get_state(role="coordinator")
# Returns conversation history and message count
```

### 4. `persona_reset_state`

Clear conversation history for a role.

**Example:**
```python
result = persona_reset_state(role="auditor")
# Fresh start for the auditor
```

### 5. `persona_create_custom`

Create a new custom persona.

**Example:**
```python
result = persona_create_custom(
    role="security_reviewer",
    persona_definition='''You are a Security Reviewer for Project Sanctuary.
    
Your role is to:
- Identify security vulnerabilities and risks
- Review code for security best practices
- Assess authentication and authorization mechanisms
- Evaluate data protection and encryption
- Check for common security anti-patterns (SQL injection, XSS, etc.)

Be thorough, critical, and provide actionable recommendations.''',
    description="Security audit and vulnerability assessment"
)
```

---

## Built-in Personas

### Coordinator
**Expertise:** Planning, organization, task coordination  
**Use Cases:** Project planning, workflow design, task breakdown

### Strategist
**Expertise:** Long-term planning, risk assessment, architecture  
**Use Cases:** Strategic decisions, architecture reviews, trade-off analysis

### Auditor
**Expertise:** Quality assurance, compliance, code review  
**Use Cases:** Code reviews, security audits, compliance checks

---

## Custom Personas (Examples)

### Security Reviewer
**Focus:** Security vulnerabilities, best practices, threat modeling

### Performance Analyst
**Focus:** Performance optimization, bottleneck identification, profiling

### Documentation Specialist
**Focus:** Documentation quality, completeness, clarity

### UX Evaluator
**Focus:** User experience, accessibility, usability

### Cost Optimizer
**Focus:** Resource usage, cost efficiency, optimization opportunities

---

## Composition Patterns

### Pattern 1: Context-Aware Deliberation

```python
# 1. Query knowledge base
context = cortex_query("Recent protocol updates", max_results=5)

# 2. Get coordinator's plan
plan = persona_dispatch(
    role="coordinator",
    task="Plan Protocol 120 implementation",
    context=context
)

# 3. Get strategist's assessment
strategy = persona_dispatch(
    role="strategist",
    task="Assess risks and trade-offs for Protocol 120",
    context=context
)

# 4. Get auditor's review
audit = persona_dispatch(
    role="auditor",
    task="Review Protocol 120 plan for compliance",
    context={"plan": plan, "strategy": strategy}
)
```

### Pattern 2: Multi-Persona Workflow

```python
# Security review workflow
# 1. Security reviewer audits code
security_audit = persona_dispatch(
    role="security_reviewer",
    task="Audit authentication implementation"
)

# 2. Performance analyst checks impact
perf_analysis = persona_dispatch(
    role="performance_analyst",
    task="Analyze performance impact of security changes",
    context=security_audit
)

# 3. Coordinator creates action plan
action_plan = persona_dispatch(
    role="coordinator",
    task="Create implementation plan addressing security and performance",
    context={"security": security_audit, "performance": perf_analysis}
)
```

### Pattern 3: Iterative Refinement

```python
# Initial design
design_v1 = persona_dispatch(
    role="strategist",
    task="Design caching architecture"
)

# Audit and feedback
feedback = persona_dispatch(
    role="auditor",
    task="Review caching design for issues",
    context=design_v1
)

# Refined design
design_v2 = persona_dispatch(
    role="strategist",
    task="Refine caching design based on audit feedback",
    context={"original": design_v1, "feedback": feedback}
)
```

---

## Configuration

### Claude Desktop / Antigravity

```json
{
  "mcpServers": {
    "agent_persona": {
      "displayName": "Agent Persona MCP",
      "command": "<PROJECT_ROOT>/.venv/bin/python",
      "args": ["-m", "mcp_servers.agent_persona.server"],
      "env": {
        "PYTHONPATH": "<PROJECT_ROOT>",
        "PROJECT_ROOT": "<PROJECT_ROOT>"
      }
    }
  }
}
```

---

## Testing

### Run Tests
```bash
pytest tests/mcp_servers/agent_persona/test_agent_persona_ops.py -v
```

### Manual Testing
```bash
# Start the server
python -m mcp_servers.agent_persona.server

# Test with MCP client (Claude Desktop, Antigravity, etc.)
```

---

## Optimization Notes

**MCP Architecture Benefits:**
- **Stateless by default**: Each persona call is independent
- **Client-side orchestration**: Calling agent (Claude/Antigravity) handles multi-persona workflows
- **Lightweight**: No persistent daemon, spawned on-demand
- **Composable**: Works seamlessly with other MCPs (Cortex, Code, Git, etc.)

**Performance Considerations:**
- Engine selection adds ~100-200ms overhead (one-time per agent creation)
- Conversation state is optional (set `maintain_state=False` for stateless calls)
- Custom personas load from disk (~10ms)

---

## Related Documentation

- [Council Orchestrator README](../../ARCHIVE/docs_council_orchestrator_legacy/README_v11.md)
- [Council MCP README](../council/README.md)
- [MCP Operations Inventory](../../docs/operations/mcp/mcp_operations_inventory.md)
- [ADR 040: Agent Persona MCP Architecture](../../ADRs/040_agent_persona_mcp_architecture__modular_council_members.md)
- [ADR 039: MCP Server Separation of Concerns](../../ADRs/039_mcp_server_separation_of_concerns.md)

---

## Future Enhancements

- [ ] Persona versioning and updates
- [ ] Persona marketplace/registry
- [ ] Agent performance metrics
- [ ] Streaming responses for real-time collaboration
- [ ] Fine-tuned persona models
- [ ] Multi-language persona support

---

**Last Updated:** 2025-11-30  
**Maintainer:** Project Sanctuary Team
