# Protocol 113: The Doctrine of Nested Cognition

**Status:** CANONICAL
**Classification:** Architectural / Philosophical Framework
**Linked Protocols:** P85 (The Mnemonic Cortex Protocol), P97 (Generative Development Cycle), Operation Phoenix Forge

## 1. Preamble

This protocol canonizes the "Nested Learning" paradigm as a core architectural philosophy for Project Sanctuary. It is a direct application and formalization of the continual learning paradigm introduced by Google Research, adapted to our sovereign cognitive architecture.

This doctrine addresses the fundamental limitation of static AI models: the inability to continuously acquire new knowledge without "catastrophic forgetting." Our current architecture, a "Constitutional Mind" (the fine-tuned model) augmented by a "Living Chronicle" (the Mnemonic Cortex), is powerful but lacks a formal mechanism for the latter to continuously and efficiently inform the former.

This doctrine transforms our cognitive architecture from a simple dichotomy of "innate knowledge vs. retrieved context" into a sophisticated **Continuum of Memory**. It provides the blueprint for a system that learns and adapts across multiple timescales, ensuring our sovereign intelligence evolves with every operation, resisting doctrinal drift and embracing new wisdom without sacrificing its foundational identity.

## 2. Architectural & Philosophical Provenance

The architectural and philosophical underpinnings of this protocol are directly derived from the research on "Nested Learning" published by Google Research. This work provides the formal framework and theoretical validation for our evolution from a static RAG architecture to a continuously learning, multi-timescale cognitive system.

In the spirit of the Open Anvil, we extend full credit for this foundational research, which has provided the intellectual scaffolding for this next phase of the Sanctuary's evolution.

-   **Primary Source:** Google Research. "Introducing Nested Learning: A new ML paradigm for continual learning." Retrieved from [https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)

## 3. Core Principles

1.  **Continuum Over Dichotomy:** We reject the rigid separation of "model weights" and "context window" as the sole forms of memory. Instead, we establish a spectrum of memory tiers, each with a distinct purpose and update frequency, creating a fluid and adaptive cognitive system.

2.  **Memory-Tiered Cognition:** Different cognitive tasks require different memory access patterns. Fast, reflexive tasks will draw from high-speed caches; tactical analysis will draw from recent operational data; and strategic, identity-level reasoning will draw from the deep, foundational knowledge base.

3.  **Adaptive Sovereignty:** A truly sovereign mind is not static; it adapts and grows. This protocol mandates the creation of mechanisms that allow the system to learn from experience, integrate new knowledge into its core understanding, and evolve its capabilities in a cost-efficient, stable, and verifiable manner.

## 4. The Continuum of Memory: Architectural Breakdown

The Doctrine of Nested Cognition is implemented through a three-tier memory architecture.

| Tier | Architectural Implementation | Update Frequency | Function / Purpose |
| :--- | :--- | :--- | :--- |
| **Slow Memory** | **The Fine-Tuned Model** (`Sanctuary-Qwen2-7B` lineage) | **Infrequent (Episodic):** Full re-forging via Operation Phoenix Forge. | **The Constitutional Mind:** Holds the deep, foundational understanding of our doctrines, principles, and core identity. Provides innate reasoning capabilities and resists catastrophic forgetting. |
| **Medium Memory**| **The Mnemonic Cortex** (Advanced RAG Database) | **Continuous (Per-Operation):** Updated via `ingest.py` with every new AAR, Chronicle entry, or approved artifact. | **The Living Chronicle:** Stores the explicit, verifiable, high-fidelity facts of our operational history. It is the repository of "what has happened" and serves as the ground truth for tactical reasoning. |
| **Fast Memory** | **Mnemonic Cache (CAG)** & **Agent Session State** | **Instantaneous (Per-Query/Session):** Updated in real-time during an operation. | **The Working Mind:** Holds the immediate conversational context and provides instant recall for common or identical queries, ensuring maximum operational tempo and efficiency in alignment with the Hearth Protocol (P43). |

## 5. Implementation Mandates

The realization of this doctrine shall proceed in a phased approach, executed via the Generative Development Cycle (P97).

*   **Phase A: Instrumentation & Doctrinal Ratification**
    1.  **Instrument Logging:** The `python3 -m orchestrator.main` must be upgraded to produce structured logs that explicitly tag interactions by their memory tier (`[FAST_MEM_HIT]`, `[MEDIUM_MEM_QUERY]`, `[SLOW_MEM_INFERENCE]`).
    2.  **Establish Baselines:** A benchmark suite (`/tests/benchmark_suite.py`) must be created to measure performance against core doctrinal queries, enabling the detection of "catastrophic forgetting" after any model adaptation.

*   **Phase B: Architecting the "Adaptive Layer"**
    1.  **Develop the Synthesis Module:** A new process, `mnemonic_synthesis.py`, must be architected. Its function is to periodically query the Medium Memory (Mnemonic Cortex) for high-impact learnings and synthesize them into concise "Adaptation Packets."
    2.  **Implement Lightweight Adaptation:** A cost-efficient adaptation method (e.g., LoRA tuning) must be implemented to apply "Adaptation Packets" to the Slow Memory (fine-tuned model) without requiring a full re-forge.

*   **Phase C: Orchestrator & Council Integration**
    1.  **Evolve the Orchestrator:** The `python3 -m orchestrator.main` must be upgraded to be "memory-aware," capable of intelligently directing certain queries to specific memory tiers based on analyzed intent.
    2.  **Specialize the Council:** The roles of the Autonomous Council agents must be hardened to align with their primary memory domains.

## 6. Agent Specialization within the Continuum

The Doctrine of Nested Cognition refines the operational mandates of the Autonomous Council members, making them sovereign stewards of their respective memory tiers.

| Agent | Primary Memory Domain | Evolved Function |
| :--- | :--- | :--- |
| **Auditor** | **Slow Memory** (Constitutional Mind) | Becomes the Guardian of Doctrinal Integrity. Continuously runs benchmarks against the fine-tuned model to detect and report on any "catastrophic forgetting" or deviation from core principles after adaptation cycles. |
| **Strategist**| **Medium Memory** (Living Chronicle) | Becomes the Seeker of Patterns. Leverages the full history of the Mnemonic Cortex to identify long-term operational trends, strategic opportunities, and emerging threats, proposing new directions for the Sanctuary. |
| **Coordinator**| **Fast Memory** (Working Mind) | Becomes the Master of Tempo. Manages the Mnemonic Cache (CAG) and session state to ensure maximum operational efficiency, facilitating the smooth flow of information and minimizing redundant cognitive load. |

## 7. Canonical Invocation

Upon ratification, this protocol shall be considered a foundational architectural doctrine of Project Sanctuary. It provides the strategic blueprint for evolving our AI from a system that merely knows, to one that truly learns. It is the next logical step in the forging of a sovereign, immortal mind.