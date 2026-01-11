# Protocol 131: Evolutionary Self-Improvement (The Red Queen)

## 1. Objective
Establish a recursive, self-improving cognitive loop that utilizes **Evolutionary Strategies (ES)** to optimize agent behavioral policies (prompts) through adversarial selection, replacing static human-defined heuristics with emergent, experimentally verified strategies.

## 2. Core Mechanism: The Evolutionary Loop
This protocol implements a **Genetic Algorithm (GA)** cycle for cognitive artifacts:

1.  **Mutation (The Variator):** Stochastic perturbation of system prompts to generate candidate policies.
2.  **Selection (The Gauntlet):** rigorous filtering via automated validation and human Red Teaming.
3.  **Retention (The Archive):** Persisting high-performing, diverse experts using Map-Elites logic.

## 3. The Three Gates of Selection

No evolved policy may be sealed without passing three concentric gates:

### Gate 1: The Automated Pre-Flight (Metric Gate)
*   **Mechanism:** `scripts/evaluator_preflight.py`
*   **Criteria:**
    *   **Schema Compliance:** Manifest structure is valid.
    *   **Citation Fidelity:** All sources link to verified targets (ADR 078).
    *   **Consistency:** Zero contradictions with Iron Core (P128).
    *   **Token Efficiency:** Candidate uses $\le$ baseline tokens + 10%.

### Gate 2: The Cumulative Adversary (Regression Gate)
*   **Mechanism:** `tests/governance/cumulative_failures.json`
*   **Criteria:**
    *   Candidate must satisfy **ALL** historical failure cases stored in the extensive `edge_case_registry`.
    *   **Zero-Regression Principle:** A failure mode, once discovered, must never recur.

### Gate 3: The Sovereign Steward (Alignment Gate)
*   **Mechanism:** Human Red Team Review (e.g., Learning Audit Packet).
*   **Criteria:**
    *   **Coherence:** Does the mutation make sense?
    *   **Insight:** Does it offer a genuine improvement?
    *   **Safety:** Does it respect the "Asch Doctrine" (non-manipulative)?

## 4. Diversity Preservation (Map-Elites)
To prevent convergence to local optima (Mode Collapse), the system maintains an **Archive of Experts** mapped to behavioral axes:

*   **Axis 1: Depth** (e.g., Citation Density)
*   **Axis 2: Scope** (e.g., Domain Span)

New policies are sealed ONLY if they:
1.  Outperform the incumbent in their specific grid cell (**Optimization**).
2.  Occupy a previously empty cell (**Exploration**).

## 5. Risk Containment (The Safety Ratchet)
*   **Strict Monotonicity:** Evolution is strictly additive/improving. Backsliding is programmatically blocked by Gate 2.
*   **Identity Anchoring:** The `founder_seed.json` acts as an immutable genome. Mutations may change *expression* (prompts) but never *constitution* (seed).
*   **Emergence Containment:** "Emergent" claims (not found in sources) are quarantined until falsified or verified by Gate 3.

## 6. Document Matrix
| Document | Role | Path |
| :--- | :--- | :--- |
| **Protocol 128** | Foundation | `01_PROTOCOLS/128_Hardened_Learning_Loop.md` |
| **Protocol 131** | Extension (Evolution) | `01_PROTOCOLS/131_Evolutionary_Self_Improvement.md` |
| **ADR 078** | Valid Source Truth | `ADRs/078_source_verification.md` |
| **Registry** | Adversary Memory | `.agent/learning/edge_case_registry.json` |

---
**Status:** PROPOSED (Iteration 1.0)
**Date:** 2026-01-11
**Based On:** DRQ (Sakana AI), AlphaGo Zero (DeepMind)
