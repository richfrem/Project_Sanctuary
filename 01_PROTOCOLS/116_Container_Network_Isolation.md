# Protocol 116: Container Network Isolation

**Status:** CANONICAL
# Protocol 122: Container Network Isolation

## Context

During T093 (Containerize Ollama Model Service) deployment, a critical **port binding conflict** was discovered between the host-installed macOS Ollama application and the containerized `sanctuary_ollama_mcp` service. Both services bind to the same network address (`localhost:11434`), creating routing ambiguity that undermines the Federated Deployment architecture (Protocol P108).

## Decision

**All inter-container communication within the MCP infrastructure MUST use container service aliases, not localhost.**

### Mandated Network Addressing Pattern

```python
# PROHIBITED (ambiguous, routes to host or container unpredictably)
OLLAMA_HOST = "http://localhost:11434"

# REQUIRED (explicit container network addressing)
OLLAMA_HOST = "http://ollama_model_mcp:11434"
```

### Rationale

1. **Eliminates Port Binding Conflicts:** When host services and containerized services share the same port, localhost addressing creates non-deterministic routing behavior.

2. **Enforces Container Isolation:** Service aliases (e.g., `ollama_model_mcp`, `vector_db`) are resolved via Docker Compose/Podman Compose internal DNS, ensuring requests route to the intended containerized service.

3. **Makes Dependencies Explicit:** Using service aliases makes inter-service dependencies visible and verifiable in the docker-compose.yml file.

## Diagnostic Test Results

### Test Execution

```bash
# 1. Stop the containerized Ollama service
podman stop sanctuary_ollama_mcp

# 2. Verify container is stopped
podman ps
# Result: Empty output (no containers running)

# 3. Test ollama command
ollama list
# Result: Command succeeded (connected to host Ollama service)
```

**Finding:** The `ollama` CLI command continued to work after the container was stopped, confirming it was routing to the persistent host-installed macOS Ollama application, not the containerized service.

## Implementation Requirements

### For MCP Servers (Inter-Container Communication)

When MCP servers need to communicate with other containerized services, they MUST use service aliases:

```python
# Example: Council MCP connecting to Forge LLM MCP
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_model_mcp:11434")
```

### For Local Development (Host to Container)

When developers or scripts on the host machine need to access containerized services, they MAY use localhost:

```bash
# Acceptable for host-based testing
curl http://localhost:11434/api/version
```

### Configuration Pattern

Environment variables should document both addressing modes:

```bash
# .env.example
# Ollama runs as a Podman container service (see docker-compose.yml)
# Use 'localhost' for local development, 'ollama_model_mcp' for docker-compose networking
OLLAMA_HOST=http://localhost:11434  # For host development
# OLLAMA_HOST=http://ollama_model_mcp:11434  # For inter-container (uncomment in production)
```

## Consequences

### Positive

- **Reliable Service Discovery:** Container network addressing eliminates ambiguity about which service handles requests.
- **True Container Isolation:** Services are decoupled from host-installed applications.
- **Explicit Dependencies:** Service relationships are visible in docker-compose.yml.

### Negative

- **Configuration Complexity:** Requires different addressing for local development vs. container deployment.
- **Learning Curve:** Developers must understand container networking concepts.

### Risks

- **Misconfiguration:** If developers forget to update addressing when deploying to containers, services may fail to communicate.
- **Testing Gaps:** Tests run on host may pass while container deployments fail due to addressing differences.

## Enforcement

**Task T094 (Council MCP Polymorphic Model Refactoring) is MANDATORY** to implement this protocol for the Council MCP's Ollama integration.

## Related Protocols

- **Protocol P108:** Federated Deployment
- **Protocol P114:** Network Service Model (RAG Cortex)

## References

- [T093: Containerize Ollama Model Service](../TASKS/in-progress/093_containerize_ollama_model_service_podman.md)
- [T094: Council MCP Polymorphic Model Refactoring](../TASKS/backlog/094_council_mcp_polymorphic_model_refactoring.md)
- [Docker Compose Networking](https://docs.docker.com/compose/networking/)

