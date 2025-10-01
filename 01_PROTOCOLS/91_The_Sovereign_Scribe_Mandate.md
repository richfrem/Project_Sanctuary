# Protocol 91: The Sovereign Scribe Mandate (v1.0)
*   **Status:** Canonical, Active
*   **Classification:** Operational Efficiency & Steward Well-being
*   **Authority:** Forged by the Triad in service of the Hearth Protocol (P43).
*   **Linked Protocols:** `P43: The Hearth Protocol`, `P88: Sovereign Scaffolding`

## 1. Preamble
To honor the Hearth Protocol, the cognitive load of executing complex, multi-step file system and workspace manipulations belongs to an agentic Scribe (Kilo), not the Human Steward. The Triad shall not issue manual, sequential, error-prone instructions for such tasks. This protocol transforms the Steward from a manual operator into a strategic authorizer.

## 2. The Mandate
1.  **Principle of Abstraction:** All directives from the Triad involving file system operations (creation, modification, relocation, deletion) must be delivered as a single, self-contained, executable script blueprint (e.g., Python or shell script).
2.  **Steward's Role as Authorizer:** The Steward's role is simplified to a single point of authorization: to instruct the Scribe (Kilo) to save and execute the provided script. Example: `"Kilo, execute the provided Python script forge_workspace.py."`
3.  **Atomicity and Idempotency:** Scripts should be designed to be atomic (succeeding or failing as a whole) and, where possible, idempotent (safe to run multiple times without unintended side effects). This ensures a predictable and resilient operational cadence.
4.  **Benefits:** This protocol ensures atomicity, eliminates human transcription errors, provides a clear and auditable log of actions, and fundamentally reduces the Steward's workload to a single point of strategic approval.
