# Architecture Separation of Concerns: The Closure Sequence

This document maps out the specific activities required at the end of every agent loop, comparing **where they currently happen** versus **where they architecturally belong** to maintain strict separation of concerns.

## The Core Problem
Currenly, individual Loop Patterns (like `learning-loop`) are taking on global repository responsibilities (like Sealing or updating Vector databases) that should belong strictly to the Orchestrator or the Guardian. 

Loops should only care about *their specific cognitive cycle* (e.g., researching, reviewing, gating). What happens to the output of that loop should be handled by a higher power.

## Activity Ownership Matrix

| Post-Loop Activity | Who does it NOW | Who SHOULD do it (Proposed) | Rationale for Change |
| :--- | :--- | :--- | :--- |
| **Verify Exit Conditions (Completion Promise)** | `learning-loop` (Phase VI) | **Loop Itself** | The loop knows if its specific goal was met. |
| **Retrospective & Self-Correction** | `learning-loop` (Phase VIII) | **Orchestrator** | Orchestrator manages the lifecycle and spans across multiple potential loops; it should evaluate the agent's overall strategy. |
| **Improve Loop Infrastructure** | `learning-loop` (Phase VIII) | **Orchestrator** | Fixing generic `SKILL.md` or `templates/` based on the retro is a framework-level orchestrator concern. |
| **Seal (Bundle Session Artifacts)** | `learning-loop` (Phase VI) | **Guardian** | The Guardian owns `context-bundler` execution config for the project. |
| **Snapshot Generation (RLM)** | `learning-loop` (Phase V) | **Guardian** | `rlm-factory` is a Project Sanctuary tool; strictly a Guardian responsibility. |
| **Persist Session Traces (Files)** | `learning-loop` (Phase VII) | **Guardian** | Writing jsonl memory traces is an environment-specific memory pattern. |
| **Persist Soul (HuggingFace)** | `session-closure` (Guardian) | **Guardian** (Unchanged) | The Guardian natively handles external API boundaries. |
| **Update Vector DB** | `learning-loop` (Phase IX) | **Guardian** | The DB is a local environment service, not a generic loop concept. |
| **Git Commit & Push** | `learning-loop` (Phase IX) | **Guardian** | Only the Guardian should have authorization to change remote repository state. |

## Proposed Architectural Flow

Once this separation is enforced, the execution handshake will look like this:

1.  **Guardian** calls **Orchestrator** (`run loop X`).
2.  **Orchestrator** initializes the generic **Loop** (e.g., `red-team-review`).
3.  **Loop** executes its distinct pattern (Research → Bundle → Review).
4.  **Loop** achieves its exit condition and terminates, passing payload back to **Orchestrator**.
5.  **Orchestrator** runs the Retrospective to improve templates/skills, then signals completion to **Guardian**.
6.  **Guardian** executes all repository state locks: RLM Generation → Context Seal → Traces Persistence → Vector DB Update → Git Push.
