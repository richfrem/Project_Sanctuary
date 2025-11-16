# TASK: Implement Protocol 113 - Council Memory Adaptor

**Status:** BACKLOG
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** Requires #017
**Related Documents:** `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`

## Context
This task is a core deliverable for the successful implementation of the Strategic Crucible Loop (Task #017), providing the long-term, 'slow memory' learning mechanism that is fed by the loop's outputs.

## Objective
Create a periodic Slow-Memory learning layer that distills stable knowledge from the Mnemonic Cortex, guided by signals from the CAG, to fine-tune a model memory adaptor.

## Deliverables
1.  **Adaptation Packet Generator:** Create a tool to convert round packets and CAG telemetry into a training curriculum.
2.  **Slow-Memory Update Mechanism:** Implement a safe, lightweight fine-tuning strategy (e.g., LoRA).
3.  **Versioned Memory Adaptor:** Establish a system for versioning, deploying, and rolling back memory adaptors.
