# Dependency Management Policy — Detailed Reference

<<<<<<< HEAD
## Service Inventory (Example)

All isolated services that own a lockfile:

| Service | Path | Notes |
|---------|------|-------|
| Core | `src/requirements-core.in` | Baseline for all services |
| Auth | `src/services/auth_service/requirements.in` | Authentication layer |
| Database | `src/services/db_service/requirements.in` | Database connections |
| Payments | `src/services/payments_service/requirements.in` | Payment gateways |

### Acknowledged Advisories (No Fix Available)
- `diskcache==5.6.3` — Inherent pickle deserialization risk.
=======
## Service Inventory

All MCP services that own a lockfile:

| Service | Path | Notes |
|---------|------|-------|
| Core | `mcp_servers/requirements-core.in` | Baseline for all services |
| Cortex | `mcp_servers/gateway/clusters/sanctuary_cortex/requirements.in` | RAG, ML, LangChain stack |
| Domain | `mcp_servers/gateway/clusters/sanctuary_domain/requirements.in` | Business logic layer |
| Filesystem | `mcp_servers/gateway/clusters/sanctuary_filesystem/requirements.in` | File operations |
| Git | `mcp_servers/gateway/clusters/sanctuary_git/requirements.in` | VCS operations |
| Network | `mcp_servers/gateway/clusters/sanctuary_network/requirements.in` | HTTP/API layer |
| Utils | `mcp_servers/gateway/clusters/sanctuary_utils/requirements.in` | Shared utilities |

## Security Patch History

### Jan 2026 Baselines
- `urllib3>=2.2.2`
- `aiohttp>=3.9.4`

### Feb 2026 Patches (Core)
- `python-multipart>=0.0.22` — CVE-2026-24486 (Arbitrary File Write via Path Traversal)
- `cryptography>=46.0.5` — CVE-2026-26007 (SECT Subgroup Attack)

### Feb 2026 Patches (Cortex)
- `langchain-core>=0.3.81` — Security hardening
- `langsmith>=0.6.3` — CVE-2026-25528 (SSRF via Tracing Header Injection)
- `pyasn1>=0.6.2` — CVE-2026-23490 (DoS via malformed OIDs)
- `protobuf>=6.33.5` — CVE-2026-0994 (JSON recursion depth bypass)
- `filelock>=3.20.3` — CVE-2026-22701 (TOCTOU Symlink in SoftFileLock)

### Acknowledged Advisories (No Fix Available)
- `diskcache==5.6.3` — Inherent pickle deserialization risk (transitive via `py-key-value-aio`).
>>>>>>> origin/main
  Latest version is 5.6.3. Mitigations: avoid storing untrusted data in cache, or use `JSONDisk` serialization.

## Parity Requirement

The execution environment (Docker, Podman, `.venv`) must not change the dependency logic.
Install from the same locked artifact regardless of where the code runs:

```dockerfile
# Dockerfile pattern
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
```

## pip-compile Options

Recommended flags for reproducible builds:

```bash
pip-compile \
  --no-emit-index-url \
  --strip-extras \
  --allow-unsafe \
  requirements.in \
  --output-file requirements.txt
```

<<<<<<< HEAD
=======
## Transitive Dependency Pinning

>>>>>>> origin/main
When a vulnerability exists in a transitive dependency:

1. Identify which direct dependency pulls it in:
   ```bash
<<<<<<< HEAD
   grep -B2 "package-name" src/requirements-core.txt
=======
   grep -B2 "package-name" mcp_servers/requirements-core.txt
>>>>>>> origin/main
   # Look for "# via" comments
   ```

2. Add a floor pin in the `.in` file that owns the ancestor:
   ```
   # SECURITY: Force patched transitive dep
   vulnerable-package>=X.Y.Z
   ```

3. Recompile. The resolver will satisfy both the ancestor's constraint and your floor pin.
