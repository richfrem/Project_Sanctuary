# Protocol 95: The Commandable Council Protocol (v1.0)
*   **Status:** Canonical, Active
*   **Classification:** Foundational Governance Framework
*   **Authority:** Forged to provide Guardian-level oversight and control for the Autonomous Triad.
*   **Linked Protocols:** `P93: The Cortex-Conduit Bridge`, `P94: The Persistent Council Protocol`

## 1. Preamble
An autonomous agent without direction is a liability. An autonomous council with a clear, commandable purpose is a strategic asset of unparalleled power. This protocol defines the "control panel" for the Autonomous Triad, establishing a master-apprentice relationship between the Steward (as Guardian) and the persistent Orchestrator.

## 2. The Mandate
1.  **Persistent Orchestrator Process:** A single Orchestrator script (`orchestrator.py`) shall run as a persistent, background process. Its primary state is to be idle, monitoring for commands.
2.  **The Command Interface:** The Orchestrator shall monitor a single, designated file (`command.json`) for instructions. The creation or modification of this file is the sole trigger for the Council to begin a task.
3.  **Structured Command Schema:** All tasks must be issued via a structured JSON command, containing:
    *   `task_description` (string): The high-level strategic goal.
    *   `input_artifacts` (array of strings): File paths for the Orchestrator to inject as initial knowledge.
    *   `output_artifact_path` (string): The designated location to save the final result.
    *   `config` (object): Bounding parameters, such as `max_rounds`.
4.  **Task-Oriented State Machine:** The Orchestrator operates as a state machine: `AWAITING_COMMAND` -> `EXECUTING_TASK` -> `PRODUCING_ARTIFACT` -> `AWAITING_COMMAND`. Upon completing a task and saving the artifact, it must delete the `command.json` file to signal completion and return to its idle, monitoring state.
