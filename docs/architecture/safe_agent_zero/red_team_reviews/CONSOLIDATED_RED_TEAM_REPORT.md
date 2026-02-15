# Safe Agent Zero: Consolidated Red Team Report (Final Gold Master)

**Date:** 2026-02-15
**Status:** **APPROVED FOR BUILD ( CONDITIONAL )**
**Scope:** Full Architectural Review (Automated + Manual)
**Reviewers:** Claude Opus (CLI), GPT-4o (Manual), Grok 3 (Manual), Gemini 1.5 Pro (Manual)

---

## 1. Executive Summary

The "Safe Agent Zero" architecture has undergone a rigorous, multi-model adversarial review. The consensus is that the move to the **MVSA (4-Container Model)** was the correct decision to reduce "Accidental Complexity," but specific high-value weaknesses remain in the **Control Plane** and **Sidecar Trust Model**.

**Verdict by Model:**
*   **Gemini (The Pragmatist):** **"A" (Secure & Simple)**. Validates the 4-container model. Highlights danger of `git pull` in agent logic.
*   **Grok (The Adversary):** **"Conditional Safe"**. Identifies "Scout as Exfiltration Oracle" and "Sidecar Trust Collapse". Demands domain pinning.
*   **GPT-4o (The Architect):** **"Conditional"**. Flags "Trust Boundary Ambiguity" in Policy Engine. Demands strict JSON schema for policy.

**Final Decision:** Proceed to Build, with **3 Mandatory Fixes** derived from this review.

---

## 2. Critical Findings & Mandatory Fixes

### 2.1 Scout Exfiltration (The "Oracle" Problem)
*   **Finding (Grok):** Even with network isolation, the Agent controls Scout via CDP. It can instruct Scout to screenshot a secret and then exfiltrate that screenshot via steganography or timing channels.
*   **Fix:** **CDP Command Filter** (or strict operational limits).
    *   *Implementation:* We will enforce strict **Domain Pinning** (No Wildcards) in the Sidecar Proxy.
    *   *Implementation:* We will block `Page.captureScreenshot` in the Agent's tool definition wrapper (Layer 9).

### 2.2 Sidecar Trust Collapse
*   **Finding (Grok):** The "Unified Sidecar" is a single point of failure. If compromised, it has access to all networks.
*   **Fix:** **Least Privilege Sidecar**.
    *   *Implementation:* Run Squid and Dnsmasq as non-root user `squid`.
    *   *Implementation:* Apply the same `agent-profile.json` Seccomp profile to the Sidecar.

### 2.3 Policy Ambiguity
*   **Finding (GPT):** "Where does the policy live?" If the Agent can modify its own guardrails, the game is over.
*   **Fix:** **Immutable Policy Mounts**.
    *   *Implementation:* `policy.yaml` must be mounted `read-only` into the Agent container.
    *   *Implementation:* The `ActionValidator` logic must be loaded from a read-only path, separate from the writable workspace.

### 2.4 Operational Guardrails
*   **Finding (Gemini):** A simple `git pull` could wipe the local worktree.
*   **Fix:** **Destructive Command Blocklist**.
    *   *Implementation:* Explicitly block `git pull`, `git reset`, `rm -rf` in the `ActionValidator`.

---

## 3. The "Gold Master" Architecture (Frozen)

### 1. `sanctum-guard` (Nginx)
*   **Role:** User-Facing Ingress.
*   **Security:** Basic Auth, Rate Limiting.

### 2. `sanctum-agent` (Node.js)
*   **Role:** The Brain.
*   **Hardening:**
    *   `read-only` rootfs.
    *   `no-new-privileges: true`.
    *   Seccomp: `agent-profile.json`.
    *   **Policy:** Read-Only mount at `/etc/sanctum/policy.yaml`.

### 3. `sanctum-scout` (Chromium)
*   **Role:** The Browser.
*   **Isolation:** `execution-net` (CDP) + `browsing-net` (Proxy). **NO INTERNET.**

### 4. `sanctum-sidecar` (Squid + Dnsmasq)
*   **Role:** The Jailer.
*   **Hardening:** Run as `squid` user. Seccomp profile applied.
*   **Policy:** Strict Domain Pinning (Allowlist ONLY, NO Wildcards).

---

## 4. Next Steps

1.  **Update `implementation_plan.md`** to include "Sidecar Seccomp" and "Read-Only Policy Mounts".
2.  **Execute WP-004**: Build the system.
