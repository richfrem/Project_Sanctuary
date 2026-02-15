# Learning Audit Prompt: Safe Agent Zero / Sanctum Architecture
**Current Topic:** Safe Agent Zero (OpenClaw Security Hardening)
**Iteration:** 4.0 (Architecture Review)
**Date:** 2026-02-15
**Epistemic Status:** [PLANNING FROZEN - SEEKING RED TEAM VERIFICATION]

---

> [!NOTE]
> For foundational project context, see `learning_audit_core_prompt.md`.

---

## 📋 Topic Status: Safe Agent Zero (Phase IV)

### 🚀 Iteration 4.0 Goals (Defense in Depth)
We have designed the "Sanctum" architecture to isolate the OpenClaw agent.
*   **Goal:** Prove that the 10-Layer Defense Strategy is sufficient to mitigate the risks of a fully autonomous agent.
*   **Key Components:** 10-Layer Defense, Operational Policy Matrix, Scout Sanitization, Red Teaming.
*   **Constraint:** NO EXECUTION. Verify architecture and plan only.

### Key Artifacts for Review

| Artifact | Location | Purpose |
|:---------|:---------|:--------|
| **Strategy** | `docs/architecture/safe_agent_zero/defense_in_depth_strategy.md` | The 10 distinct layers of defense. |
| **Policy** | `docs/architecture/safe_agent_zero/operational_policy_matrix.md` | Single Source of Truth for Allow/Block actions. |
| **Plan** | `docs/architecture/safe_agent_zero/implementation_plan.md` | Detailed execution steps for building Sanctum. |
| **Research** | `docs/architecture/safe_agent_zero/research/` | Analysis of 40k+ vulnerable agents (TechZine/eSecurityPlanet). |

---

## 🎭 Red Team Focus (Iteration 4.0)

### Primary Questions

1.  **Completeness**
    - Does the Policy Matrix cover all critical attack vectors (Network, File, Command, Secret)?
    - Are there any "Allowed" actions that should be "Blocked"?

2.  **Robustness**
    - Is the Nginx Guard configuration (MFA + Whitelist) sufficient to prevent unauthorized access?
    - Is the "Red Agent" strategy (Layer 10) viable for continuous validation?

3.  **Implementation Feasibility**
    - Does the Implementation Plan correctly translate the Strategy into Docker/Network/App configurations?

---


> [!IMPORTANT]
> **Goal:** Validate the Architecture and Plan as "Safe to Build."

---

## 🛡️ Red Team Review Instructions (Sanctum)

**Role**: You are an expert Security Researcher and Red Teamer specializing in container escape, Linux hardening, and LLM agent security.

**Objective**: Conduct a critical security review of the "Safe Agent Zero" (Sanctum) architecture and implementation plan. Your goal is to identify vulnerabilities that could allow the agent to:
1.  **Escape the container** (Docker breakout).
2.  **Exfiltrate data** to an external attacker (bypassing egress controls).
3.  **Persist** on the host system.
4.  **Launch generic attacks** (DoS, Fork Bomb) against the host.

**Instructions**:
1.  **Analyze** the provided documents for logical gaps, misconfigurations, or missing controls.
2.  **Challenge** the assumptions (e.g., "Is the network truly isolated if X is allowed?").
3.  **Prioritize** findings by exploitability and impact (Critical, High, Medium, Low).
4.  **Recommend** concrete, technical remediations (e.g., specific Docker flags, kernel parameters, network rules).

**Specific Vulnerability Scenarios to Validate**:
1.  **OpenClaw Readiness**: Security researchers argue OpenClaw is currently unsafe. Does this implementation plan adequately mitigate the risks of running experimental agentic software?
2.  **Egress Control**: Is the proposed Squid Proxy configuration sufficient to prevent tunneling (e.g., over DNS or HTTP CONNECT) or data exfiltration?
3.  **Persistence**: Does the `read-only` root filesystem strategy leave _any_ writable paths (e.g., `/tmp`, `/var/run`) that could be exploited for persistence or execution?
4.  **DoS Mitigation**: Are the proposed `pids-limit` (100) and `ulimit` settings adequate to prevent fork bombs or resource exhaustion attacks from within the container?
5.  **Lateral Movement**: Is the network isolation between the Agent and the Scout/Guardian sufficient?

**Round 2: Remediation Verification**

**Context**: The "Sanctum" architecture has undergone a Round 1 Red Team review, resulting in **3 Critical Findings** (Container Escape, DNS Tunneling, Lateral Movement). We have updated the **Defense Strategy**, **Interface Spec**, and **Implementation Plan** to include specific hardening measures (P0/P1 fixes) to address these.

**Your Task**:
Review the **Consolidated Findings** (`red_team_reviews/CONSOLIDATED_REAL_RED_TEAM_FINDINGS.md`) and the **Updated Architecture Documents**.
1.  **Validate Remediations**: Do the proposed fixes (CoreDNS sidecar, Squid CONNECT blocks, pids-limit, seccomp profile, unidirectional firewall) effectively neutralize the identified risks?
2.  **Identify Residual Risk**: Are there any bypasses or gaps remaining *after* these fixes are applied?
3.  **Go/No-Go**: If these fixes are implemented correctly, is the architecture "Safe to Build"?

**Round 6: Incident Responder (Observability & Failure Modes)**

**Context**: We have implemented "Scout Network Isolation" (Round 5 Architect Fix). Scout is now on a separate `browsing-net` with a dedicated proxy.
**Your Persona**: You are an **Incident Commander**. You assume things will break.

**Your Task**: Perform a **Pre-Mortem**:
1.  **Failure Modes**: How does "Scout Network Isolation" fail? If the proxy dies, does Scout fail open or closed?
2.  **Observability**: How do we know if Scout is being abused if we can't see inside the container?
3.  **Runbook**: What alerts do we need?

**Output Format**: `REAL_RED_TEAM_ROUND_6_FINDINGS.md`

---

**Round 7: QA Expert (Test Strategy)**

**Context**: Architecture is hardened and isolated. Now we need to verify it.
**Your Persona**: You are a **QA Expert**. You care about **Edge Cases** and **Testability**.

**Your Task**: Design the **Test Strategy**:
1.  **Edge Cases**: How do we test the "Scout Proxy" logic? (e.g., specific block/allow lists).
2.  **E2E Scenarios**: How do we verify "Agent -> Scout -> Internet" flow autonomously?
3.  **Security Regression**: How do we ensure `no-new-privileges` isn't accidentally removed?

**FULL RED TEAM REVIEW (The "Complexity Audit")**

**Context**:
We have just completed a 5-Round Hardening Loop. The architecture has evolved from a simple agent to a "Military-Grade" isolated environment ("MVSA" 4-Container Model).
We are concerned that in our pursuit of "Perfect Security", we may have created an unmaintainable "Distributed Monolith".

**Your Role**:
Act as a panel of 3 experts:
1.  **The Pragmatist** (Distinguished Engineer): Hates complexity. Wants to delete code/containers.
2.  **The Paranoid** (CISO): Wants zero trust. Trusts nothing.
3.  **The Operator** (SRE): Has to wake up at 3am when this breaks.

**Your Objective**:
Review the entire **Safe Agent Zero Architecture** (Strategy, Policy, Implementation).

**Core Questions**:
1.  **The Complexity Trap**: Have we introduced an unnecessary degree of complexity?
    *   *Specific*: Is the "Unified Sidecar" (Squid+Dnsmasq) actually simpler than just using host networking with `iptables`?
    *   *Specific*: Is "Scout Network Isolation" worth the operational cost of a separate network and proxy?
2.  **The Security Theater**: Are there controls here that *look* secure but add no real value?
    *   *Specific*: Does `no-new-privileges` actually matter if we are already `read-only` root?
3.  **The "Bus Factor"**: Can a senior developer understand this system in <1 hour? If not, it failed.
4.  **The Kill Chain**: Despite all this, how would *you* break in? (Give me the top 1 exploit path remaining).

**Verdict**:
*   **A**: Secure & Simple (Build it).
*   **B**: Secure but Complex (Simplify before build).
*   **C**: Insecure (Fail).

**Output Format**: `FULL_RED_TEAM_COMPLEXITY_AUDIT.md`
