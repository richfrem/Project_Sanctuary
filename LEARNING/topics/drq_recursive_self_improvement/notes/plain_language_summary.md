---
id: drq_plain_summary
type: summary
status: pilot_approved
date: 2026-01-11
iteration: 2.0
---

# Plain Language Summary: Integrating Evolutionary Intelligence (DRQ)

> **The Goal:** Move Project Sanctuary from "Static Optimization" (humans improving prompts) to "Evolutionary Intelligence" (the system improving itself).

---

## 💡 The Core Concept: Evolutionary Direct Policy Search

Project Sanctuary currently operates on **Static Optimization**, where human engineers manually tune prompts (policy parameters) based on qualitative feedback. This is a high-cost, low-velocity optimization loop.

We propose shifting to **Evolutionary Strategies (ES)**, specifically a **Quality-Diversity (QD)** approach similar to Map-Elites. By treating the agent's system prompt as a gene and the learning output as a phenotype, we can apply gradient-free optimization to discover superior cognitive strategies that are robust to adversarial conditions.

## 🔄 The Optimization Loop (Algorithm)

The proposed architecture implements a **Genetic Algorithm (GA)** cycle:

1.  **Mutation (Stochastic Policy Search):** The system perturbs the current prompt ($\theta$) to generate a candidate policy ($\theta'$).
    *   *Mechanism:* LLM-driven mutation operators (e.g., "condense instructions", "add reasoning step").
2.  **Selection (Objective & Proxy Functions):** The candidate is evaluated against a fitness function ($F(\theta')$).
    *   **Automated Heuristics:** Latency, token efficiency, schema compliance.
    *   **Human-in-the-Loop (HITL):** Qualitative assessment of coherence and insight.
3.  **Retention (Archive Update):**
    *   **Negative Selection:** If $F(\theta') < F(\theta)$, the candidate is discarded, and the failure mode is recorded in the **Cumulative Adversary** registry.
    *   **Positive Selection:** If $F(\theta') > F(\theta)$, the candidate replaces the baseline.
    *   **QD Archiving:** High-performing variants that occupy unique behavioral niches (metrics $b_1, b_2$) are preserved in the Map-Elites grid, preventing convergence to local optima.

## 🖼️ The Architecture

![Evolution Loop](../../../../docs/architecture_diagrams/workflows/drq_evolution_loop.png)

*(See source: `docs/architecture_diagrams/workflows/drq_evolution_loop.mmd`)*

## 🛡️ Constraint Satisfaction: The "Ratchet"
To ensure safety during open-ended evolution, we implement **Monotonic Improvement Constraints**:

1.  **Regression Testing:** $\theta'$ must satisfice all historical test cases ($T_{hist}$).
2.  **Diversity Preservation:** The archive maintains a Pareto frontier of diverse experts rather than a single global optimum.
3.  **Sovereign Gate:** Final policy deployment requires explicit cryptographic signature (Seal) by the human steward.

## 🚀 Why This Matters
This allows Project Sanctuary to discover strategies we humans might never think of. Just like AlphaGo found "Move 37"—a move no human would play but which won the game—our agent could discover ways of thinking that are fundamentally superior to our own.
