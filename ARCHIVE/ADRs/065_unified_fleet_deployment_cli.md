# Unified Fleet Operations Makefile ("The Iron Makefile")

**Status:** accepted
**Date:** 2025-12-20
**Author:** Grok (xAI), based on Red Team Analysis and Best Practices  

## Context

Building on the ACCEPTED v1.2 ADR, which adopted a Makefile as the unified interface for managing Project Sanctuary's "Fleet of 8" containers, this v1.3 proposal incorporates feedback from ongoing Red Team reviews and industry best practices.

**Infrastructure Foundation:**
The fleet is explicitly defined in the existing Root-Level **[`docker-compose.yml`](../docker-compose.yml)**. This YAML file remains the Source of Truth for container definitions (images, ports, volumes, networks). The proposed Makefile acts solely as the *operational interface* to this existing definition, ensuring valid orchestration sequences.

**Key Motivations for Iteration:**
- **User Feedback on .env and Readability:** v1.3 adds native .env sourcing in Make for parity with python logic.
- **Modularity for Client Scripts:** Extracting `wait_for_pulse.sh` for reuse.
- **Best Practices Integration:**
  - Emphasize declarative targets for build/test/deploy.
  - Add support for dynamic subsets (e.g., restart specific containers).
  - Enhance observability with logs and exec targets.
  - Improve health checks with configurable retries/timeouts.
- **Addressing Remaining Risks:** Strengthen idempotency checks and state reconciliation.

This maintains the rejection of a full Python wrapper due to complexity, while making the Makefile more feature-rich and user-friendly.

## Decision (v1.3)

We propose evolving the Root-Level `Makefile` to include enhanced targets, .env integration, and modular helpers. The Makefile remains the "single source of truth" for repeatability, with no runtime deps beyond standard tools (Make, sh, Podman).

### Design Principles

1. **Transparency:** Chain shell commands visibly; echo each step for observability.
2. **Idempotency:** Leverage Podman Compose's built-in idempotency (referencing `docker-compose.yml`); add pre-checks to skip unnecessary actions.
3. **Standardization:** "Make is the API." Extend to support environments (e.g., `make up ENV=dev`).
4. **Modularity:** Extract reusable shell helpers (e.g., `wait_for_pulse.sh`).
5. **Security and Reliability:** Source .env securely; add retries/backoff; warn on state drift.

### Command Specification

The `Makefile` will support these targets (new/updated in **bold**):

* **`make up [ENV=prod] [--force]`**:
  1. Source `.env`.
  2. Check Gateway health.
  3. `podman compose -f docker-compose.yml up -d [--build if --force]` (Physical Deploy).
  4. `scripts/wait_for_pulse.sh` (Health Check).
  5. `python3 mcp_servers/gateway/fleet_orchestrator.py` (Logical Registration).
  6. **Reconcile state:** Compare `podman ps` vs. Gateway registry; warn/log drifts.

* **`make down`**:
  1. Deregister via orchestrator (if supported).
  2. `podman compose -f docker-compose.yml down [--volumes if --force-clean]`.

* **`make restart [TARGET=container-name]`**:
  1. **Dynamic subsets:** Restart all or specific service defined in `docker-compose.yml`.
  2. `make down [TARGET]` && `make up`.

* **`make status`**:
  1. `podman ps --filter "name=sanctuary"` (table format).
  2. `curl` Gateway health/registrations.
  3. **Enhanced output:** Include last heartbeat, tool counts from `fleet_registry.json`.

* **`make verify`**:
  1. Run Tier 3 connectivity tests.
  2. **New:** Integrate with monitoring.

* **New Targets for Best Practices:**
  - **`make build`** : `podman compose -f docker-compose.yml build`.
  - **`make logs [TARGET=container-name]`** : `podman compose logs -f [TARGET]`.
  - **`make exec [TARGET=container-name]`** : `podman compose exec [TARGET] /bin/sh`.
  - **`make clean`** : `podman compose down -v --rmi all`.

### Helper Scripts (Expanded)

- **`scripts/wait_for_pulse.sh`** : Enhanced loop with retries/backoff.
- **New: `scripts/check_drift.sh`** : Compare Podman state vs. Gateway registry.

## Consequences

**Positive:**
- **Improved Repeatability:** Matches `docker-compose.yml` definitions strictly.
- **Modularity:** Helpers reduce duplication.
- **Robustness:** Retries, drift detection align with SRE best practices.
- **Observability:** Verbose output, logs targets.
- **Security:** Tokens stay in env; no subprocess risks.

**Negative:**
- **Platform Dependency:** Requires `make`.

This v1.3 proposal refines v1.2 for better alignment with user needs and best practices, explicitly anchoring operations to the existing `docker-compose.yml`.


---

**Status Update (2025-12-20):** Fleet deployment fully implemented. All 8 containers deployed via Makefile, 6 logic servers registered and federating 84 tools to Gateway. Pagination issue resolved in gateway_client.py.
