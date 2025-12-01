# Agent Persona MCP Server

**Version:** 1.0.0  
**Status:** Production Ready  
**Purpose:** Configurable AI agent personas via Model Context Protocol

---

## Overview

The Agent Persona MCP provides access to specialized AI agent personas that can be used for domain-specific tasks. Each persona has a unique role, expertise, and reasoning style.

**Key Features:**
- Built-in personas (Coordinator, Strategist, Auditor)
- Custom persona support (Security Reviewer, Performance Analyst, etc.)
- Conversation state management
- Flexible engine selection (Gemini, OpenAI, Ollama)
- Model-specific targeting (e.g., Sanctuary-Qwen2-7B, GPT-4, Gemini-2.5-Pro)

## Terminology Mapping

To align with standard industry practices, we use the following terminology mapping:

| Standard Term | Legacy Sanctuary Term | Description |
|---------------|----------------------|-------------|
| **LLM Client** | Substrate | The interface to the underlying Language Model (e.g., OpenAI, Ollama) |
| **Model Provider** | Cognitive Engine | The specific provider implementation (e.g., GeminiEngine, OllamaEngine) |
| **System Prompt** | Awakening Seed | The core instruction set that defines the agent's persona |
| **Agent Config** | Core Essence | The configuration parameters defining the agent's role and behavior |
| **Orchestrator** | Council | The system that manages multiple agents and their interactions |

---

## Architecture

### Design Pattern: Configurable Service

The Agent Persona MCP follows the **"Configurable Service"** pattern - similar to Code MCP or Config MCP, but for AI personas instead of files.

**Key Components:**
1. **Persona Files** (`personas/*.txt`) - Define agent behavior and expertise
2. **State Management** (`state/*.json`) - Maintain conversation history
3. **Engine Integration** - Uses council_orchestrator's engine selection
4. **PersonaAgent Class** - Reuses proven agent implementation

### Integration with Council Orchestrator

```
Agent Persona MCP (Lightweight)
├── Persona Definitions (*.txt files)
├── State Management (conversation history)
└── Engine Selection (delegates to council_orchestrator)
    ├── select_engine(config) → Healthy engine instance
    └── PersonaAgent(engine, persona_file, state_file)
```

---

## Available Tools

### 1. `persona_dispatch`

Execute a task using a specific persona agent.

**Signature:**
```python
persona_dispatch(
    role: str,                          # Persona role
    task: str,                          # Task to execute
    context: dict | None = None,        # Optional context
    maintain_state: bool = True,        # Persist conversation
    engine: str | None = None,          # "gemini" | "openai" | "ollama"
    model_name: str | None = None,      # Specific model variant
    custom_persona_file: str | None = None
) -> dict
```

**Example - Built-in Persona:**
```python
result = persona_dispatch(
    role="auditor",
    task="Review the test coverage for the Git MCP server"
)
```

**Example - With Context from Cortex:**
```python
# 1. Get context from Cortex MCP
context = cortex_query(query="Previous security audits", max_results=3)

# 2. Dispatch to persona with context
result = persona_dispatch(
    role="security_reviewer",
    task="Audit the authentication flow",
    context=context
)
```

**Example - Specific Model:**
```python
result = persona_dispatch(
    role="coordinator",
    task="Plan the Q1 2025 roadmap",
    engine="ollama",
    model_name="Sanctuary-Qwen2-7B:latest"
)
```

### 2. `persona_list_roles`

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

- [Council Orchestrator README](../../council_orchestrator/README.md)
- [Council MCP README](../council/README.md)
- [MCP Operations Inventory](../../docs/mcp/mcp_operations_inventory.md)
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
