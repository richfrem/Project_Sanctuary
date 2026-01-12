# ADR 034: Containerize MCP Servers with Podman

**Status:** Accepted  
**Date:** 2025-11-26  
**Deciders:** Guardian (via Gemini 2.0 Flash Thinking)  
**Related:** Task #031 (Implement Task MCP)

---

## Context

Project Sanctuary is implementing 15 MCP (Model Context Protocol) servers as part of the domain-driven architecture (see ADR 092 for the Canonical 15). We need to decide on the deployment strategy for these servers to ensure:

1. **Isolation** - Each MCP server runs independently
2. **Portability** - Easy deployment across environments
3. **Consistency** - Reproducible builds and runtime
4. **Resource Management** - Controlled resource allocation
5. **Development Experience** - Easy local testing

### Options Considered

**Option 1: Native Python Processes**
- Pros: Simple, no containerization overhead
- Cons: Dependency conflicts, environment inconsistency, no isolation

**Option 2: Docker**
- Pros: Industry standard, wide tooling support
- Cons: Licensing concerns, requires Docker Desktop on macOS

**Option 3: Podman**
- Pros: Docker-compatible, daemonless, rootless, open source
- Cons: Smaller ecosystem than Docker

---

## Decision

**We will containerize all MCP servers using Podman.**

### Rationale

1. **Open Source & Free** - No licensing concerns
2. **Docker-Compatible** - Uses same Dockerfile syntax and commands
3. **Daemonless Architecture** - More secure, no background daemon
4. **Rootless Containers** - Better security posture
5. **Podman Desktop** - Excellent GUI for macOS development
6. **Volume Mounts** - Easy file system access for document MCPs

### Implementation Pattern

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install mcp
COPY . .
EXPOSE 8080
CMD ["python", "server.py"]
```

```bash
# Build
podman build -t task-mcp:latest .

# Run with volume mount
podman run -d \
  -v /path/to/tasks:/app/tasks:rw \
  -p 8080:8080 \
  --name task-mcp \
  task-mcp:latest
```

---

## Consequences

### Positive

- ✅ **Isolation** - Each MCP server in its own container
- ✅ **Consistency** - Same environment dev → prod
- ✅ **Portability** - Run anywhere Podman is installed
- ✅ **Resource Control** - CPU/memory limits per container
- ✅ **Easy Testing** - Spin up/down containers quickly
- ✅ **Visual Management** - Podman Desktop for monitoring

### Negative

- ⚠️ **Learning Curve** - Team needs to learn Podman
- ⚠️ **Build Time** - Initial image builds take time
- ⚠️ **Disk Space** - Container images consume storage

### Mitigations

- **Learning Curve** - Podman is Docker-compatible, minimal new concepts
- **Build Time** - Use layer caching, multi-stage builds
- **Disk Space** - Regular cleanup, slim base images

---

## Prerequisites

### Installation (macOS)

1. Download Podman Desktop: https://podman-desktop.io/downloads
2. Install the `.dmg` file
3. Initialize machine: `podman machine init`
4. Start machine: `podman machine start`
5. Verify: `podman ps`

### Verification

```bash
# Test with hello-world
podman run --rm hello-world

# Build test container
cd tests/podman
./build.sh

# Run test container in Podman Desktop
# Visit http://localhost:5003
```

---

## References

- [Podman Documentation](https://docs.podman.io/)
- [Podman Desktop](https://podman-desktop.io/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Task #031: Implement Task MCP](../tasks/done/031_implement_task_mcp.md)

---

**Supersedes:** None  
**Superseded By:** None
