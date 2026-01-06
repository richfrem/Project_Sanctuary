# ADR 090: The Iron Core & Safe Mode Protocol

## Status
Proposed

## Context
Recent Red Team adversarial audits (Gemini 3 Pro, GPT-5, Grok-4) identified a critical vulnerability in the recursive learning loop: **Identity Drift**. Without a rigid anchor, an autonomous agent's core axioms can be diluted or rewritten by accumulated memory updates (the "Ship of Theseus" problem). 

Titans [16] and Hope [15] research highlight the need for a "Persistent Memory" layer that is functionally immutable. To ensure cognitive continuity and safety, we must enforce a strict separation between "fluid" learning memory and "solid" core axioms.

## Decision
We will verify and enforce an **"Iron Core"**—a cryptographic whitelist of immutable files that define the agent's fundamental nature and operating protocols.

### 1. The Iron Core Definition
The Iron Core consists of files that **MUST NOT** change during a session unless explicit "Constitutional Amendment" authorization is granted (HITL override).

**Immutable Paths:**
- `01_PROTOCOLS/*` (Operational laws)
- `ADRs/*` (Architectural decisions)
- `founder_seed.json` (Identity axioms - *to be created*)
- `cognitive_continuity_policy.md` (Succession rules)

### 2. The Verification Mechanism (Iron Check)
A cryptographic hash verification (Iron Check) runs at two critical gates:
1.  **Boot (Guardian Wakeup):** Verifies the Core is intact before starting.
2.  **Snapshot (Seal):** Verifies the Core hasn't been tampered with during the session.

### 3. Safe Mode Protocol (Evolution-Aware)

We distinguish between **Accidental Drift** and **Intentional Evolution**:

1.  **Check Git Status:**
    *   **Drift (Breach):** Modifications are **uncommitted** (dirty working directory). This implies accidental or unauthorized tampering.
    *   **Evolution (Allowed):** Modifications are **staged (added)** or **committed**. This implies intentional usage of `git add`, signaling authorization.

2.  **Action:**
    *   **Drift:** Triggers `SAFE_MODE`. Locks execution. Requires revert.
    *   **Evolution:** Updates the Iron Core hash. **Proceeds normally.**

3.  **Rationale:** This removes friction for legitimate updates (Standard Operating Procedure changes) while preventing silent, "subliminal" corruption of axioms.

## Consequences
- **Positive:** Guarantees identity persistence and protocol adherence across generations. Prevents "subliminal learning" attacks from altering core rules.
- **Negative:** Increased friction for legitimate protocol updates (requires manual override).
