# Standardization of Local Container Orchestration Tooling

**Status:** proposed
**Date:** 2025-12-16
**Author:** Antigravity & User


---

## Context

Project Sanctuary has standardized on Podman (Protocol 116) for its container runtime. We are now implementing the MCP Gateway (Task 116) which requires multi-container orchestration (Gateway + DB + Redis).
We need to decide how to process `compose.yml` files locally.
Options:
1. `docker-compose` with Podman socket emulation (requires active socket/service).
2. `podman-compose` (daemons-less translation to podman CLI args).

The user's environment is macOS with Podman Desktop. We experienced missing `docker-compose` binary issues.

## Decision

We will standardize on **`podman-compose`** for local development orchestration.

1. **Tooling:** We will add `podman-compose` to the project's development dependencies (`dev` group in `pyproject.toml` or `uv` usage).
2. **Configuration:** We will maintain `podman-compose.yml` (or standard `docker-compose.yml` verified to work with podman-compose).
3. **Workflow:** Developers will use `podman-compose up` for local stack bring-up.

This aligns with our "Unbreakable" philosophy by removing the fragile dependency on a background system socket/daemon.

## Consequences

**Positive:**
- **Zero Daemon Dependency:** `podman-compose` works with rootless Podman directly, removing the need for `podman system service` or a running socket.
- **Lightweight:** Python-based, easy to install in the project venv.
- **Pod Native:** Better translation of compose concepts to Podman pods.

**Negative:**
- **Feature Gap:** Does not support 100% of the Docker Compose v3.9 spec (though sufficient for our needs).
- **Tooling Divergence:** Developers familiar with `docker compose up` need to Type `podman-compose up`.

**Mitigation:**
- We will stick to the subset of Compose spec that is fully supported.
- We will provide `make` targets to abstract the specific command.
