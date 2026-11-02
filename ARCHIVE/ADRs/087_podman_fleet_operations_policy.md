# ADR 087: Podman Fleet Operations Policy

**Status:** APPROVED
**Date:** 2026-01-01
**Author:** Sanctuary Guardian

---

## CONTEXT
The Project Sanctuary fleet runs on Podman via `docker-compose.yml`. Because all services share a common build context (`context: .`), a change to *any* file in the root directory technically invalidates the build cache for all services.
- **Problem:** Running `podman compose up -d --build` rebuilds the entire fleet (8+ containers) even if only one (e.g., `rag_cortex`) was modified.
- **Impact:** Wasted time (3-5 minutes per cycle) and unnecessary downtime.

## DECISION

### 1. Mandate Targeted Rebuilds
Operatives must explicitly target the service they modified when running build commands.

**Correct Pattern:**
```bash
podman compose -f docker-compose.yml up -d --build sanctuary_cortex
```

**Forbidden Pattern:**
```bash
podman compose -f docker-compose.yml up -d --build
```
*(Unless core shared libraries in `sanctuary_domain` or `gateway` have changed).*

### 2. Fleet Registry Refresh
After any restart, the Gateway's dynamic registry must be refreshed to ensure 100% routing accuracy.

**Procedure:**
1.  Wait for Healthcheck (Port 8000/810X active).
2.  Run: `python3 mcp_servers/gateway/fleet_setup.py`

### 3. Pruning Hygiene
To prevent disk exhaustion from frequent rebuilds:
```bash
podman system prune -f
```

## CONSEQUENCES
- **Positive:** Reduces iteration loop from ~5 mins to ~45 seconds.
- **Positive:** Reduces risk of accidental regressions in untouched services.
- **Negative:** Requires operative discipline to remember service names.
