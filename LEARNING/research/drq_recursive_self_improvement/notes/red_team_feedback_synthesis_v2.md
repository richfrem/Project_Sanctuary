---
id: drq_red_team_synthesis_v2
type: audit_response
status: active
date: 2026-01-11
iteration: 2.0
---

# Red Team Feedback Synthesis: DRQ Application (Iteration 2.0)

> **Verdict:** 🟢 **CONDITIONAL APPROVAL** (Doctrine Sealed, Machinery Pending)
> **Summary:** The architectural "Doctrine" (Protocol 131, P128 v4.0) is **APPROVED**. The "Machinery" (Evaluator, Registry, Metrics) is **MISSING**.
> **Next Phase:** Implementation Constraints (Pilot).

---

## 🛡️ The Sealed Doctrine
The following artifacts are now stable and approved "Rules of the Road":
1.  `01_PROTOCOLS/131_Evolutionary_Self_Improvement.md`
2.  `plugins/guardian-onboarding/resources/protocols/128_Hardened_Learning_Loop.md` (v4.0 Branch)
3.  `docs/architecture_diagrams/workflows/drq_evolution_loop.png`
4.  `edge_case_registry.json` (Concept Approved)

## 🛠️ Implementation Mandates (The "Engine" Build)

### 1. Gate 1: The Automated Evaluator (`evaluator_preflight.py`)
*   **Must Check:**
    *   **Citation Fidelity:** Detect 404s/Reference integrity (ADR 078).
    *   **Schema:** Valid JSON/Manifest compliance.
    *   **Efficiency:** Token usage vs baseline.
*   **Constraint:** Logic must be **symbolic/deterministic**, NOT "LLM-as-Judge" (to avoid circular bias).

### 2. Gate 2: The Cumulative Adversary (`edge_case_registry.json`)
*   **Structure:**
    ```json
    { "topic": "drq", "cases": [ { "id": "001", "check": "citation_density > 0.5" } ] }
    ```
*   **Constraint:** Zero-Regression Principle. Once added, never removed.

### 3. Map-Elites Metrics (Post-Hoc Compute)
*   **Constraint:** NEVER self-reported by LLM. Must be `def measure_depth(text) -> float`.
*   **Depth Proxy:** `citation_density` + `token_complexity`.
*   **Scope Proxy:** `file_touch_count` + `domain_span` (graph distance).

### 4. Pilot Target: `learning_audit_prompts.md`
*   **Constraint:** Do NOT touch `guardian_prompt`.
*   **Goal:** Use the new loop to optimize the audit prompt itself.

---

## 🚀 The Build Plan (Sprint 1)
1.  **Scaffold:** Create `scripts/evaluator_preflight.py`.
2.  **Seed:** Create `.agent/learning/edge_case_registry.json` with Iteration 1.0 feedback.
3.  **Metric:** Implement `measure_depth` draft function.
4.  **Test:** Run against the current packet.
