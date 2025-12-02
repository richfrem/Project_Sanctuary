# Council MCP Refactoring Notes (Task 60268594)

## Bootstrap Requirements

When the Council MCP server starts or handles its first request, it should initialize the system:

### 1. Cache Initialization (via Cortex MCP)

```python
from mcp_servers.lib.cortex.operations import CortexOperations

cortex = CortexOperations()

# Check if cache is populated
stats = cortex.cache_stats()

if stats["hot_cache_size"] == 0:
    # Warm up cache with genesis queries
    cortex.cache_warmup()
    
# Optional: Generate Guardian boot digest
cortex.guardian_wakeup()
```

### 2. Deliberation Logic Refactoring

The core deliberation logic from `ARCHIVE/council_orchestrator_legacy/orchestrator/app.py` needs to be refactored to:

1. **Use Agent Persona MCP** instead of internal agents:
   ```python
   from mcp_servers.lib.agent_persona.agent_persona_ops import AgentPersonaOperations
   
   persona_ops = AgentPersonaOperations()
   
   # For each agent in deliberation
   response = persona_ops.dispatch(
       role="strategist",  # or coordinator, auditor
       task=task_description,
       context=context_from_previous_rounds,
       model_name="Sanctuary-Qwen2-7B:latest"
   )
   ```

2. **Use Cortex MCP** for memory/RAG:
   ```python
   # Query knowledge base
   results = cortex.query(query, max_results=5)
   ```

3. **Use Round Packets** for tracking:
   ```python
   from mcp_servers.lib.council.packets.schema import CouncilRoundPacket
   
   packet = CouncilRoundPacket(
       timestamp=...,
       session_id=...,
       round_id=...,
       member_id=agent_role,
       decision=response["response"],
       # ... other fields
   )
   ```

### 3. Startup Hook Pattern

Add a lazy initialization pattern to `council_ops.py`:

```python
class CouncilOperations:
    def __init__(self):
        self._initialized = False
        self.cortex = None
        self.persona_ops = None
    
    def _ensure_initialized(self):
        if not self._initialized:
            self._bootstrap()
            self._initialized = True
    
    def _bootstrap(self):
        """Initialize system on first use"""
        from mcp_servers.lib.cortex.operations import CortexOperations
        from mcp_servers.lib.agent_persona.agent_persona_ops import AgentPersonaOperations
        
        self.cortex = CortexOperations()
        self.persona_ops = AgentPersonaOperations()
        
        # Warm up cache if needed
        stats = self.cortex.cache_stats()
        if stats["hot_cache_size"] == 0:
            logger.info("Cache empty, warming up...")
            self.cortex.cache_warmup()
    
    def dispatch_task(self, task_description, ...):
        self._ensure_initialized()
        # ... deliberation logic
```

## Migration Checklist

- [ ] Add bootstrap logic with cache warmup
- [ ] Refactor deliberation to use Agent Persona MCP
- [ ] Integrate Cortex MCP for memory queries
- [ ] Use packets/ system for round tracking
- [ ] Remove dependency on archived orchestrator
- [ ] Update tests to reflect new architecture
- [ ] Update README with new bootstrap behavior

## References

- Legacy orchestrator: `ARCHIVE/council_orchestrator_legacy/orchestrator/app.py`
- Round packets: `mcp_servers/lib/council/packets/`
- Agent Persona MCP: `mcp_servers/lib/agent_persona/`
- Cortex MCP: `mcp_servers/lib/cortex/`
