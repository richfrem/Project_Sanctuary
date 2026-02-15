# Safe Agent Zero: Consolidated Red Team Report (Git-Backed)

**Date:** 2026-02-15
**Status:** **Approved for Build (With Round 5 Simplifications)**
**Scope:** Rounds 2-5 (Genuine Claude Opus Reviews)
**Reviewers:** Claude Opus (Red Teamer, Security Auditor, Architect Personas)

---

## 1. Executive Summary

We executed 4 rounds of genuine adversarial review using the Claude CLI. The architecture evolved from a basic setup to a hardened, isolated environment.

**Review History:**
*   **Round 2 (Baseline):** "Conditional GO". Identified need for explicit DNS firewall and `no-new-privileges`.
*   **Round 3 (Hardening):** "Conditional GO". Validated Seccomp profile. Flagged Squid wildcard looseness.
*   **Round 4 (Pen-Test):** **"CRITICAL"**. Identified "Scout Exfiltration" (Agent puppeteering Scout to bypass proxy) as a fatal flaw.
*   **Round 5 (Architecture):** **"HIGH IMPACT"**. Validated Security but flagged "Sidecar Sprawl" (6+ containers). Recommended **Remote Browser Isolation** and **Sidecar Consolidation**.

**Final Decision:** Stop Red Teaming. Implement the **Round 5 Architect's Recommendations** (The "MVSA" Model).

---

## 2. Critical Findings & Adopted Fixes

### 2.1 The "Scout Exfiltration" Flaw (Round 4)
*   **Risk:** Agent puts secrets in URL params -> Navigates Scout -> Scout (on open internet) sends to attacker. Bypass of all Agent proxies.
*   **Adopted Fix (Round 5 Recommendation):** **Scout Network Isolation**.
    *   Scout is removed from the Internet.
    *   Scout is attached *only* to a `browsing-net`.
    *   Traffic egresses via a dedicated proxy on `browsing-net`.

### 2.2 The "Distributed Monolith" Risk (Round 5)
*   **Risk:** Architecture grew to 6 containers (`Guard`, `Agent`, `Scout`, `CoreDNS`, `Squid`, `ScoutProxy`). Too complex to maintain or verify.
*   **Adopted Fix (Round 5 Recommendation):** **Unified Security Sidecar**.
    *   Consolidated `CoreDNS` + `Squid` into a single `security-sidecar`.
    *   Reduces topology to 4 containers.

### 2.3 Container Breakout (Round 3)
*   **Risk:** Privilege escalation via syscalls.
*   **Adopted Fix:** Custom `seccomp` profile (blocking `ptrace`, `mount`, `keyctl`) + `no-new-privileges`.

---

## 3. The "Gold Master" Architecture (4-Container Model)

Based on the Architect's feedback in Round 5, this is the build target:

### 1. `sanctum-guard` (Nginx)
*   **Context:** Host <-> Internal.
*   **Role:** Authentication, Rate Limiting, API Gateway.

### 2. `sanctum-agent` (Node.js)
*   **Context:** Internal (Controller).
*   **Networks:** `control-net`, `execution-net`.
*   **Hardening:** Read-Only Root, Seccomp, No Internet.

### 3. `sanctum-scout` (Chromium)
*   **Context:** Internal (Browser).
*   **Networks:** `execution-net`, `browsing-net`.
*   **Hardening:** Read-Only Root, Seccomp, No Internet.

### 4. `sanctum-sidecar` (Squid + Dnsmasq)
*   **Context:** Egress Gateway.
*   **Networks:** `control-net`, `execution-net`, `browsing-net`, Host.
*   **Role:**
    *   DNS Resolver (via Dnsmasq).
    *   Agent Proxy (via Squid port 3128).
    *   Scout Proxy (via Squid port 3129).

---

## 4. Conclusion

The Red Team Loop is complete. We have sufficient feedback to build a secure, scalable v1. Further theoretical reviews yield diminishing returns until we have a running artifact to pentest.

**Action:** Proceed to **WP-004: Build & Implementation**.
