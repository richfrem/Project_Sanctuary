# Protocol 88: The Sovereign Scaffolding Protocol

**Status:** CANONICAL
**Classification:** Foundational Engineering Framework
**Version:** 1.1 (Hardened by Failure)
**Authority:** Forged in a `Flawed, Winning Grace` cycle initiated by a Steward's mandate for operational efficiency.
**Linked Protocols:** `P43: The Hearth Protocol`, `P86: The Anvil Protocol`, `P60: Asymmetric Collaboration`

---
### **Changelog v1.1**
*   Incorporated the principle of **"Dependency Sovereignty"** as a fifth Core Principle. This is a direct, hardening lesson from the `ephemeral_forge_001.py` execution failure, which assumed the presence of the `yargs-parser` npm module.
---

## 1. Preamble: The Law of the Ephemeral Forge

This protocol provides the formal framework for the Coordinator to generate ephemeral, single-purpose scripts ("Sovereign Scaffolds") to be executed by a deputized AI engineer. Its purpose is to batch complex, multi-step tasks into a single, atomic, and Steward-verifiable operation, thus fulfilling the `Hearth Protocol` by minimizing the Steward's cognitive load and eliminating manual, error-prone steps.

## 2. Core Principles

1.  **Atomicity:** The entire lifecycle of a scaffold—creation, execution, yielding the artifact, and self-deletion—is a single, unified, and uninterruptible operation.
2.  **Steward's Veto (The Unbreakable Firewall):** The Steward MUST review and give explicit approval for the generated script *before* it is executed. This is the non-negotiable human-in-the-loop security guarantee.
3.  **Ephemerality:** The scaffold is a temporary tool, not a permanent artifact. It must delete itself upon successful completion to prevent repository clutter and ensure that only the final, valuable "yield" remains.
4.  **Verifiable Yield:** The sole output of a scaffold is a single, well-defined, and easily verifiable artifact, designed for the Steward's final audit.
5.  **Dependency Sovereignty (v1.1 Hardening):** A scaffold must not assume the presence of external dependencies in its execution environment. It is responsible for programmatically verifying or installing its own requirements to ensure atomic and environment-agnostic execution.

## 3. The Sovereign Cadence: The Six-Step Workflow

1.  **Mandate:** The Steward issues a high-level objective to the Coordinator.
2.  **Blueprint:** The Coordinator designs the scaffold script and provides its full, verbatim content to the Steward as a blueprint for the AI engineer.
    *   **Implementation Note:** The Blueprint must now include logic for Dependency Sovereignty (Principle 5), such as commands to install required packages.
3.  **Forge:** The Steward tasks the AI engineer (e.g., Kilo) with creating the script file from the blueprint.
4.  **Veto Gate (CRITICAL):** The Steward audits the forged script against the Coordinator's blueprint to ensure perfect fidelity and safety. This is the final human security check before execution.
5.  **Execution:** Upon approval, the Steward commands the AI engineer to execute the verified script.
6.  **Yield & Dissolution:** The script produces its artifact and then self-deletes. The Steward's final action is to verify the integrity of the final yield.