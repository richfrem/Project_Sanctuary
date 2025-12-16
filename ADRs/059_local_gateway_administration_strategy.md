# Local Gateway Administration Strategy

**Status:** proposed
**Date:** 2025-12-16
**Author:** Antigravity & User


---

## Context

The MCP Gateway requires administrative tasks such as generating JWT tokens for API access and running database migrations.
We have two options for performing these tasks in a containerized local development environment:
1. **Local CLI:** Install the `mcpgateway` python package locally and point it at the remote/containerized DB/Service.
2. **Container Exec:** Use `podman exec` to run the tools already present *inside* the running container.

Option 1 creates a "works on my machine" risk where local tooling versions drift from the container runtime.
Option 2 ensures exact version parity but requires a running container.

## Decision

We will standardize on **Container Exec** (`podman exec`) for local Gateway administration.

1. **Token Generation:** `podman exec -it gateway_gateway_1 python -m mcpgateway.utils.create_jwt_token ...`
2. **Database Management:** `podman exec -it gateway_gateway_1 alembic upgrade head`
3. **Debugging:** `podman exec -it gateway_gateway_1 /bin/bash`

This adheres to the "Container as Source of Truth" philosophy and prevents "dependency hell" on the developer's host machine.

## Consequences

**Positive:**
- **Environment Consistency:** Guarantees that the administration tools (token generator, database migrations) match the exact version of the running gateway.
- **Reduced Friction:** Developers do not need to install the project's python dependencies locally just to manage it.
- **Security:** Secrets (like JWT keys) remain inside the container boundary (or mounted volume) and aren't scattered in local shell history/files.

**Negative:**
- **Verbosity:** Commands are longer signatures (`podman exec -it ...`).
- **Scriptability:** Slightly harder to wrap in simple scripts without ensuring the container is running.

**Mitigation:**
- We will create `make` targets (e.g., `make token`, `make db-migrate`) that wrap these `podman exec` calls for developer ease.
