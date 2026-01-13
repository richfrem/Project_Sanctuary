# ADR 092: RLM-Based Context Synthesis (The Cognitive Hologram)

**Status:** Proposed
**Date:** 2026-01-12
**Author:** Cortex Guardian
**Protocol:** 132

## Context
Project Sanctuary relies on `learning_package_snapshot.md` to transfer context between sessions. Currently, this is a "Diff" (showing only recent changes).
This leads to "Context Blindness" where the agent knows *what changed* but forgets the *fundamental architecture* (the "Dark Matter" problem).

## Decision
We will replace the "Diff-based Snapshot" with a **Recursive Language Model (RLM) Synthesis**.
1.  **Mechanism:** Upon sealing, an RLM agent will recursively summarize the *entire* relevant state (Protocols + ADRs + Active Code).
2.  **Artifact:** The `learning_package_snapshot.md` becomes a "Cognitive Hologram"—a high-fidelity, compressed map of the *entire* system state.
3.  **Tooling:** We will build `cortex_rlm_synthesize` to automate this "MapReduce" logic at the end of every loop.

## Consequences
*   **Positive:** "Wakeup Hallucinations" (guessing architecture) should drop to near zero. Agents wake up "knowing" the system.
*   **Negative:** Sealing time increases (from 5s to ~60s). Cost per seal increases (RLM tokens).
*   **Mitigation:** Use "Lazy Hashing"—only re-summarize modules that have changed hash since the last seal.

## Compliance
*   **Iron Core:** This creates a *derived* artifact. It does not modify the Iron Core itself.
*   **Protocol 128:** Inserts a new step (Phase V) before the final Git Commit.
