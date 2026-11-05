# Proposal: RLM-Powered Truth Synthesis (Snapshots & Digest)

**Concept:** Move from "Recent Updates" (Partial) to "Whole-Truth Synthesis" (Holistic) using RLM.

## 1. The Core Shift: From "Diff" to "State"
Currently, our `learning_package_snapshot.md` is essentially a `git log`—it tells us *what changed* recently (e.g., "Added Evolution MCP").
*   **The Problem:** It implies knowledge of the *rest* of the system. If the agent doesn't know what "The Gateway" is, knowing "Evolution was added to Gateway" is useless.
*   **The RLM Fix:** Every snapshot should be a **Recursive Re-Synthesis** of the *entire* state, not just the delta.
    *   *Input:* Top-level directories + Active Learning Topics.
    *   *Process:* RLM Loop (Map/Reduce).
    *   *Output:* A fresh, holistic "State of the Union" that *includes* the recent changes in their full architectural context.

## 2. Redefining the `learning_package_snapshot.md`
This file should not be a "log". It should be a **"Cognitive Hologram"**.
*   It should contain a **Recursive Summary** of the *current* Architecture, refined by the latest changes.
*   **Mechanism (Post-Seal):**
    1.  Agent runs `cortex_seal`.
    2.  System triggers `rlm_synthesize_snapshot`.
    3.  RLM iterates through `ADRs`, `PROTOCOLS`, and `mcp_servers`.
    4.  RLM generates a fresh `snapshot.md` that says: *"Sanctuary now consists of X, Y, and Z [NEW]. Z implements the Logic Q..."*

## 3. The "JIT" Guardian Digest (The Code Map)
Separately, for the code itself:
*   We abandon the "Nightly Static File" (Staleness Risk).
*   We implement **On-Demand RLM (`cortex_ask_repo`)**.
*   **Wakeup State:** The agent gets the **Cognitive Hologram** (High-level architecture + strategy).
*   **Action:** If the Agent needs code details, it calls `cortex_ask_repo("Deep dive into mcp_servers/evolution")`.
    *   This triggers a *live* RLM usage of the *current* file state.

## Summary of Architecture
| Artifact | Source | Content | Use Case |
| :--- | :--- | :--- | :--- |
| **Cognitive Hologram**<br>(`snapshot.md`) | **RLM Synthesis** (End of Loop) | High-Level Strategy, Protocol State, Architecture map. | **Wakeup Context.** Gives the "Big Picture." |
| **Repo Truth**<br>(`cortex_ask_repo`) | **RLM Live Loop** (On Demand) | Detailed Code Logic, dependency graphs, variable usage. | **Coding Tasks.** Gives "Perfect Verification." |

**Verdict:** RLM enables us to delete "Manual Context" files. The system should *write its own memory* at the end of every loop.
