# Protocol 93: The Cortex-Conduit Bridge (v1.0)
*   **Status:** Canonical, Conceptually Active
*   **Classification:** Agentic Knowledge Framework
*   **Authority:** Forged to prevent agentic amnesia and ensure all actions are grounded in truth.
*   **Linked Protocols:** `P85: Mnemonic Cortex`, `P92: Mnemonic Conduit Protocol`, `P95: The Commandable Council`

## 1. Preamble
An autonomous agent's power is proportional to the depth of its context. An agent operating without memory is a mere tool; an agent grounded in the totality of its history is a true cognitive partner. This protocol establishes the architectural bridge between an acting agent and our living memory.

## 2. The Mandate
1.  **The Orchestrator as Proxy:** Agents shall not have direct access to the file system or Mnemonic Cortex. The Orchestrator (per P95) serves as the agent's sovereign proxy for all knowledge retrieval.
2.  **The Formal Request Token:** An agent must formally request knowledge by embedding a machine-readable token in its response. The canonical token is: `[ORCHESTRATOR_REQUEST: ACTION(parameter)]`.
    *   *Example:* `[ORCHESTRATOR_REQUEST: READ_FILE(PROMPTS/00_framework-overview.md)]`
    *   *Example:* `[ORCHESTRATOR_REQUEST: QUERY_CORTEX("Find all protocols related to agentic safety")]`
3.  **The Fulfillment Loop:** The Orchestrator must parse agent responses for these tokens. Upon detection, it must:
    a. Fulfill the request (e.g., read the file, run the RAG query).
    b. Inject the resulting knowledge as context into the prompt for the next agent in the dialogue sequence.
    c. Log its action in the final artifact for auditability.


