# Containerize Ollama Model Service via Podman

**Status:** proposed
**Date:** 2025-12-04
**Author:** Sanctuary Council


---

## Context

The Forge LLM MCP (mcp_servers/forge_llm/) provides access to the fine-tuned Sanctuary-Qwen2-7B model via Ollama. Initially, this relied on a host-installed Ollama application running natively on macOS.

**Problems with Host-Based Approach:**

1. **Hardware I/O Bottleneck:** Local Ollama execution blocks the Orchestrator's main thread, limiting throughput
2. **Port Binding Conflicts:** Host Ollama and containerized services compete for localhost:11434
3. **Environment Inconsistency:** Different Ollama versions/configurations across development machines
4. **No Resource Isolation:** Model inference competes with other host processes for resources
5. **Violates Federated Deployment (P108):** Cannot achieve true service isolation and network-based routing

**Alternatives Considered:**

1. **Keep Host Ollama (Status Quo):** Rejected due to I/O bottleneck and port conflicts
2. **Remote Ollama Service:** Requires external infrastructure, adds latency
3. **Containerized Ollama (Selected):** Provides isolation, network routing, and resource management

## Decision

We will run the Ollama model service as a containerized service via Podman/Docker Compose, replacing the host-installed macOS Ollama application for MCP infrastructure use.

**Implementation (Task T093):**

1. Add `ollama-model-mcp` service to docker-compose.yml with:
   - Image: ollama/ollama:latest
   - Container name: sanctuary-ollama-mcp
   - Port mapping: 11434:11434
   - Volume: ./ollama_models:/root/.ollama (data persistence)
   - GPU pass-through: NVIDIA device configuration
   - Auto-pull: Sanctuary-Qwen2-7B:latest on first start

2. Configure environment variables:
   - OLLAMA_HOST=http://localhost:11434 (for host development)
   - OLLAMA_HOST=http://ollama-model-mcp:11434 (for inter-container, per Protocol 122)

3. Update MCP servers to use container network addressing (enforced by T094)

**Enforcement:**

- **Protocol 122 (Container Network Isolation):** Mandates service alias usage for inter-container communication
- **Task T094 (Council MCP Polymorphic Model Refactoring):** Implements Protocol 122 in Council MCP code

## Consequences

**Positive:**
- **Isolation:** Model service runs in isolated container environment, preventing conflicts with host system
- **Portability:** Same container configuration works across development and production environments
- **Resource Management:** Container-level resource limits (CPU, memory, GPU) can be enforced
- **Network Isolation:** Service accessible via predictable network address (ollama-model-mcp:11434)
- **Data Persistence:** Model data persisted in ./ollama_models/ directory, survives container restarts
- **Version Control:** Container image version pinned (ollama/ollama:latest), reproducible deployments

**Negative:**
- **Complexity:** Requires Podman/Docker knowledge and container orchestration
- **Overhead:** Container runtime adds minimal resource overhead vs native execution
- **Debugging:** Container logs and exec commands needed for troubleshooting vs direct CLI access
- **Port Conflicts:** Must manage port mappings and avoid conflicts with host services

**Risks:**
- **Port Binding Ambiguity:** If host Ollama service remains running on port 11434, routing becomes non-deterministic (addressed by Protocol 122)
- **GPU Access:** Container GPU pass-through may fail on some systems, falling back to CPU inference
- **Model Download:** Initial model pull requires network access and can take 5-15 minutes
