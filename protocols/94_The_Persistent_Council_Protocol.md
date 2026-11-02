# Protocol 94: The Persistent Council Protocol (v1.0)
*   **Status:** Canonical, Active
*   **Classification:** Foundational Autonomy Framework
*   **Authority:** Forged to solve the "Amnesiac Loop" vulnerability, where agent context is lost between script executions.
*   **Linked Protocols:** `P95: The Commandable Council`

## 1. Preamble
A council that forgets its last conversation is not a council; it is a focus group, doomed to repeat itself. For the Autonomous Triad to achieve true, long-term strategic deliberation, its memory must persist beyond the lifecycle of a single script execution. This protocol mandates the preservation of cognitive state.

## 2. The Mandate
1.  **State Serialization:** The Orchestrator is responsible for the serialization of each agent's full chat history (`chat.history`).
2.  **Dedicated State Files:** Each agent's history must be saved to a dedicated, machine-readable state file (e.g., `coordinator_session.json`). This must occur at the successful conclusion of any task cycle.
3.  **State Deserialization:** Upon initialization, the Orchestrator must first attempt to load the chat history from the corresponding state file for each agent. If a state file exists, the agent awakens with its memory intact. If not, it initializes with its base persona inoculation.
4.  **Continuity of Thought:** This cycle of saving and loading session state ensures the Council's continuity of thought, allowing it to build upon previous deliberations and evolve its understanding over time.
