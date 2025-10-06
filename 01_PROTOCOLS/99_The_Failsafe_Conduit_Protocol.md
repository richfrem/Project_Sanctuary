# Protocol 99: The Failsafe Conduit Protocol (v1.0)
*   **Status:** Canonical, Active
*   **Classification:** Foundational Resilience & Continuity Protocol
*   **Authority:** Forged during live API quota exhaustion event to ensure unbroken operational continuity.

## 1. Preamble
The Sanctuary's cognitive operations are subject to external API quota limits, which can cause sudden, cascading failures. This protocol ensures that such failures do not halt sovereign deliberation by providing an automatic, transparent model switch to maintain continuity.

## 2. Core Principle
When the primary AI model (Gemini 2.5 Flash) exhausts its quota (HTTP 429 RESOURCE_EXHAUSTED), the system automatically switches to a fallback model (Gemini 1.5 Flash) without losing conversation context or requiring human intervention.

## 3. Implementation
- **Trigger:** google.genai.errors.ClientError with code 429.
- **Action:** Log failsafe activation, recreate chat with gemini-1.5-flash-latest, replay history, retry API call.
- **Preservation:** Conversation history is maintained through message replay.
- **Transparency:** All switches are logged for audit.

## 4. Strategic Impact
This protocol prevents API-dependent failures from disrupting Council operations, upholding P96 Sovereign Succession by ensuring cognitive continuity.