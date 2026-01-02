# ADR 086: Empirical Epistemic Gating (The Edison Mandate)

**Status:** APPROVED
**Date:** 2026-01-01
**Author:** Gemini (Red Team Lead)
**Context:** Round 3 Audit of Multi-Model Collaboration

---

## 1. The Variance Paradox
High-coherence latent collaboration (LatentMAS) creates a "Borg" risk: models converge on consensus so quickly they bypass independent verification.
- **Problem:** Low variance = High Efficiency but High Pathological Risk (Asch Conformity).
- **Metric:** We rely on **Semantic Entropy (SE)** as a proxy for epistemic state.

## 2. Decision: Dynamic Cognitive Coupling
We reject a static architecture in favor of a **Dynamic Coupling** model controlled by an "Edison Breaker" in `operations.py`.

| State | SE Range | Coupling Mode | Protocol |
|-------|----------|---------------|----------|
| **Rigidity** | 0.0 - 0.2 | **DECOUPLED** | Force `search_web` / External Audit |
| **Flow** | 0.3 - 0.7 | **COUPLED** | LatentMAS / Dense Communication |
| **Chaos** | 0.8 - 1.0 | **DECOUPLED** | Force `reasoning_mode="thorough"` |

## 3. The Dead-Man's Switch
The `persist_soul` operation must implement a **Fail-Closed** logic:
1.  Calculate SE of the snapshot.
2.  If SE < 0.2 (Rigidity): **QUARANTINE** (Stop "Zombie Knowledge").
3.  If SE > 0.8 (Hallucination): **QUARANTINE**.
4.  If Exception: **QUARANTINE** (Assign SE=1.0).

## 4. Epistemic Scars
To prevent "Legibility Collapse" (where discarded possibilities are erased), the system MUST persist **Counterfactuals** alongside the final decision.
- **Mechanism:** `red_team_feedback_round_X.md` files must be included in the Seal.

## 5. Consequences
- **Positive:** Prevents "Mode Collapse" in long-term model lineages.
- **Negative:** Rejects roughly 20% of "valid" but efficient optimizations.
