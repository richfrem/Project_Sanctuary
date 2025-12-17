# Decouple IBM Gateway to External Podman Service

**Status:** accepted
**Date:** 2025-12-16
**Author:** AI Assistant
**Supersedes:** ADR 057 (Partial), Strategy 060 (Vendoring)

---

## Context

Project Sanctuary previously considered "Strategy 060" (Vendor-in-Repo) for the IBM ContextForge gateway to maintain code visibility and control. However, internal integration caused significant build friction, dependency conflicts, and CI stability issues (CodeQL alerts).

We have pivoted to **Strategy 058: Decoupling**. We will treat the Gateway as an external, "Black Box" service running in a container via Podman.

**The Strategic Trade-off:**
While decoupling solves the repository bloat issue (-1.2M tokens, clean git history), it introduces a critical **Opacity Risk**. We can no longer easily inspect the code we are running. We must mitigate this with a "Red Team" security posture adapted for external services.

## Decision

We will decouple the IBM ContextForge Gateway from the `Project_Sanctuary` repository entirely. It will run as a standalone service managed by Podman.

### Red Team Analysis & Safeguards

To compensate for the loss of code visibility, we implement a **Triple-Layer Defense** doctrine designed for "Black Box" services.

#### 1. The "Enemy" Profile (Risk Assessment)
*   **Opacity Risk:** We are executing foreign code handling sensitive LLM prompts without continuous static analysis (CodeQL) on that code.
*   **Configuration Drift:** The external state (container image, config volumes) may evolve differently than the `Project_Sanctuary` client expects, leading to silent failures or security gaps.
*   **Zombie State:** Long-running containers may accumulate compromised state or memory leaks.

#### 2. Triple-Layer Defense (Security Mandates)

**Layer 1: Network Isolation (The Air Gap)**
*   The Gateway must strictly bind to `127.0.0.1` (Localhost).
*   **Mandate:** It must **NEVER** be exposed to `0.0.0.0` or any external interface.
*   **Mechanism:** Docker/Podman port mapping must be explicit: `-p 127.0.0.1:8080:8080`.

**Layer 2: Authentication Circuit Breaker**
*   We enforce a simplified but strict authentication barrier.
*   **Mandate:** The Gateway must reject *any* connection lacking the specific Bearer token.
*   **Mechanism:** A hardcoded `MCPGATEWAY_BEARER_TOKEN` (e.g., `sanctuary-admin-key-2025`) in the `.env` file. If the header is missing or incorrect, the connection is dropped immediately.

**Layer 3: The "Clean Slate" Protocol**
*   To prevent "Zombie State" and configuration drift, we forbid persistent container lifecycles for the gateway.
*   **Mandate:** "Restarting" a simplified container is prohibited.
*   **Mechanism:** Administration requires a `podman rm -f` followed by a fresh `podman run`. Configuration is re-injected fresh on every boot.

### Vendor Refresh Protocol (Maintenance)

We treat the external gateway code as a frozen artifact that is manually updated only during specific "Refresh Events".

**Protocol:**
1.  **Manual Pull:** Updates only occur when a human admin navigates to the external `sanctuary-gateway` directory and runs `git pull`.
2.  **Rebuild:** The container image `sanctuary-gateway:latest` is strictly rebuilt (`make container-build`) immediately after a pull.
3.  **Verification:** The `scripts/verify_gateway_integrity.py` (future work) script is run against the new build before it handles production traffic.

## Consequences

**Positive:**
*   **Repository Hygiene:** Project Sanctuary is cleaner, smaller, and focused on core logic.
*   **Separation of Concerns:** Gateway operations are distinct from Agent operations.
*   **Security:** The "Clean Slate" protocol reduces the attack surface of long-running processes.

**Negative:**
*   **Operational Friction:** Requires manual container management (Podman) outside the standard flow.
*   **Blind Spots:** We lose the ability to "Jump to Definition" into gateway code during debugging.

**Compliance:**
This decision strictly adheres to Protocol 101 (Systems Safety) by enforcing isolation and explicit authentication for external components.
