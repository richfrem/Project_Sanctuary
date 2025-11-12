# TASK: Implement Protocol 113 - Council Memory Adaptor

**Status:** BACKLOG
**Priority:** Medium
**Lead:** Unassigned
**Related Documents:** `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`

## Objective
Create a periodic Slow-Memory learning layer that distills stable knowledge from the Mnemonic Cortex, guided by signals from the CAG, to fine-tune a model memory adaptor.

## Deliverables
1.  **Adaptation Packet Generator:** Create a tool to convert round packets and CAG telemetry into a training curriculum.
2.  **Slow-Memory Update Mechanism:** Implement a safe, lightweight fine-tuning strategy (e.g., LoRA).
3.  **Versioned Memory Adaptor:** Establish a system for versioning, deploying, and rolling back memory adaptors.
