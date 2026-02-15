---
name: dependency-management
description: >
  Python dependency and environment management for Project Sanctuary's MCP server fleet.
  Use when: (1) adding, upgrading, or removing a Python package, (2) responding to Dependabot
  or security vulnerability alerts (GHSA/CVE), (3) creating a new MCP service that needs its
  own requirements files, (4) debugging pip install failures or Docker build issues related
  to dependencies, (5) reviewing or auditing the dependency tree, (6) running pip-compile.
  Enforces the pip-compile locked-file workflow and tiered dependency hierarchy.
---

# Dependency Management

## Core Rules

1. **Never `pip install <pkg>` directly.** All changes flow through `.in` → `pip-compile` → `.txt`.
2. **Always commit both `.in` and `.txt` together.** The `.in` is human intent; the `.txt` is the machine-verified lockfile.
3. **One runtime per service.** Each MCP service owns its own `requirements.txt` lockfile.

## Repository Layout

```
mcp_servers/
├── requirements-core.in          # Tier 1: shared baseline (fastapi, pydantic, mcp…)
├── requirements-core.txt         # Lockfile for core
├── gateway/clusters/
│   ├── sanctuary_cortex/
│   │   ├── requirements.in       # Tier 2: inherits core + heavy ML deps
│   │   └── requirements.txt
│   ├── sanctuary_domain/
│   │   ├── requirements.in
│   │   └── requirements.txt
│   ├── sanctuary_filesystem/
│   │   ├── requirements.in
│   │   └── requirements.txt
│   ├── sanctuary_git/
│   │   ├── requirements.in
│   │   └── requirements.txt
│   ├── sanctuary_network/
│   │   ├── requirements.in
│   │   └── requirements.txt
│   └── sanctuary_utils/
│       ├── requirements.in
│       └── requirements.txt
```

## Tiered Hierarchy

| Tier | Scope | File | Examples |
|------|-------|------|----------|
| **1 – Core** | Shared by >80% of services | `requirements-core.in` | `fastapi`, `pydantic`, `fastmcp`, `httpx` |
| **2 – Specialized** | Service-specific heavyweights | `<service>/requirements.in` | `chromadb`, `langchain`, `sentence-transformers` |
| **3 – Dev tools** | Never in production containers | `requirements-dev.in` | `pytest`, `black`, `ruff` |

Each service `.in` file begins with `-r ../../../requirements-core.in` to inherit the core.

## Workflow: Adding or Upgrading a Package

1. **Declare** — Add or update the version constraint in the correct `.in` file.
   - If the package is needed by most services → `requirements-core.in`
   - If only one service → that service's `.in`
   - Security floor pins use `>=` syntax: `cryptography>=46.0.5`

2. **Lock** — Compile the lockfile:
   ```bash
   # Core
   pip-compile mcp_servers/requirements-core.in \
     --output-file mcp_servers/requirements-core.txt

   # Individual service (example: cortex)
   pip-compile mcp_servers/gateway/clusters/sanctuary_cortex/requirements.in \
     --output-file mcp_servers/gateway/clusters/sanctuary_cortex/requirements.txt
   ```
   Because services inherit core via `-r`, recompiling a service also picks up core changes.

3. **Sync** — Install locally to verify:
   ```bash
   pip install -r mcp_servers/gateway/clusters/<service>/requirements.txt
   ```

4. **Verify** — Rebuild the affected Podman container to confirm stable builds.

5. **Commit** — Stage and commit **both** `.in` and `.txt` files together.

## Workflow: Responding to Dependabot / Security Alerts

1. **Identify the affected package and fixed version** from the advisory (GHSA/CVE).

2. **Determine tier placement:**
   - Check if the package is a **direct** dependency (appears in an `.in` file).
   - If it only appears in `.txt` files, it's **transitive** — pinned by something upstream.

3. **For direct dependencies:** Bump the version floor in the relevant `.in` file.
   ```
   # SECURITY PATCHES (Mon YYYY)
   package-name>=X.Y.Z
   ```

4. **For transitive dependencies:** Add a version floor pin in the appropriate `.in` file
   to force the resolver to pull the patched version, even though it's not a direct dependency.

5. **Recompile all affected lockfiles.** Since services inherit core, a core change means
   recompiling every service lockfile. Use this compilation order:
   ```bash
   # 1. Core first
   pip-compile mcp_servers/requirements-core.in \
     --output-file mcp_servers/requirements-core.txt

   # 2. Then each service
   for svc in sanctuary_cortex sanctuary_domain sanctuary_filesystem \
              sanctuary_git sanctuary_network sanctuary_utils; do
     pip-compile "mcp_servers/gateway/clusters/${svc}/requirements.in" \
       --output-file "mcp_servers/gateway/clusters/${svc}/requirements.txt"
   done
   ```

6. **Verify the patched version appears** in all affected `.txt` files:
   ```bash
   grep -i "package-name" mcp_servers/requirements-core.txt \
     mcp_servers/gateway/clusters/*/requirements.txt
   ```

7. **If no newer version exists** (e.g., inherent design risk like pickle deserialization),
   document the advisory acknowledgement as a comment in the `.in` file and note mitigations.

## Container / Dockerfile Constraints

- Dockerfiles **only** use `COPY requirements.txt` + `RUN pip install -r requirements.txt`.
- No `RUN pip install <pkg>` commands. No manual installs.
- Copy `requirements.txt` **before** source code to preserve Docker layer caching.

## Common Pitfalls

- **Forgetting to recompile downstream services** after a core `.in` change.
- **Pinning `==` instead of `>=`** for security floors — use `>=` so `pip-compile` can resolve freely.
- **Adding dev tools to production `.in` files** — keep `pytest`, `ruff`, etc. in `requirements-dev.in`.
- **Committing `.txt` without `.in`** — always commit them as a pair.

## 🎯 Agent Diagnostics (The Dependency Doctor)
If `pip-compile` fails:
1.  **Check for Conflict**: Read the error. Is `package A` requiring `lib<2.0` while `package B` needs `lib>=3.0`?
    *   **Action**: Check if `package A` has a newer version. Bump it in `.in`.
2.  **Circular Dependency**:
    *   **Action**: Temporarily comment out the circular ref in `.in`, compile, then uncomment.
3.  **Environment Mismatch**:
    *   **Action**: Run `pip list` to ensure your local env isn't polluting the build. Use a fresh `venv` if unsure.
