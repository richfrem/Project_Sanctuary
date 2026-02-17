---
description: Audit the dependency tree for conflicts, stale pins, or security issues
---

# Audit Dependencies

## Check for Conflicts
```bash
# Dry-run compile to surface resolver conflicts
pip-compile --dry-run mcp_servers/requirements-core.in
```

## Find Stale Pins
```bash
# Show outdated packages
pip list --outdated --format=columns
```

## Verify Security Patches Applied
```bash
# Check specific package version across all lockfiles
grep -i "<package>" mcp_servers/requirements-core.txt \
  mcp_servers/gateway/clusters/*/requirements.txt
```

## Verify Dockerfile Compliance
Dockerfiles should ONLY use:
```dockerfile
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
```
**No `RUN pip install <pkg>` allowed.**
