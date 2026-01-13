# Risk Mitigation & Sanctuary Mapping (Red Team Response)

**Status:** Iteration 2.1 (Addressing Final Red Team Feedback)
**Reviewers:** Gemini, ChatGPT, Grok, Claude

---

## 1. Safety & Risk Mitigation (The "Runaway Loop")

**Concern:** Recursion introduces infinite loop risks and cost explosions.
**Mitigation Strategy (Protocol 128 Amendment):**

| Risk | Mitigation / Guardrail | Implementation |
| :--- | :--- | :--- |
| **Infinite Recursion** | **Depth Limiter** | Hard cap `MAX_DEPTH = 3` in any RLM loop. |
| **Cost Explosion** | **Budgeting** | `MAX_TOTAL_TOKENS` per session. "Early Exit" logic if confidence > 95%. |
| **Drift/Hallucination** | **Sandwich Validation** | Root Agent must re-verify the aggregated summary against a random sample of chunks. |

**Sanctuary Policy:** Any "Deep Loop" tool MUST have a `budget` parameter exposed to the Caller.

---

## 2. Sanctuary Architecture Mapping (Canonical)

**Concern:** Explicitly map RLM components to Sanctuary Protocols to prevent successor hallucination.

| External Concept | Sanctuary Component | Integration Point | Constraint |
| :--- | :--- | :--- | :--- |
| **DeepMind Titans** | **Mnemonic Cortex** | Future: Neural Memory weights. | Requires external "Surprise Metric" gating. |
| **RLM Loop** | **Protocol 128 (IV. Audit)** | `cortex_learning_debrief` (Scout) -> RLM (Deep Reader). | Must be Bounded (Depth=3). |
| **Recursive Steps** | **Protocol 113 (Council)** | Specialized sub-prompts (e.g., "Auditor") via `code_execution`. | No autonomous state mutation. |
| **Context Variable** | **Soul Traces / Ephemeral** | Intermediate summaries become `soul_traces.jsonl` entries. | Never sealed as "Truth" until synthesized. |
| **Long Context** | **Ephemeral Workspace** | The raw 10MB file in RAM. | Volatile; lost on session end. |

---

## 3. Known Failure Modes & Non-Guarantees (Pre-Mortem)

> **CRITICAL:** Recursive self-correction is **not guaranteed** to converge and must be externally gated.

1.  **"The Telephone Game" (Semantic Drift):** Summaries of summaries lose critical nuance.
    *   *Fix:* Keep "Key Quotes" in every summary layer (pass reference citations up the chain).
2.  **"Fractal Hallucination" (Optimism Bias):** A small error in Chunk 1 is amplified by the Root Agent because it is "internally coherent."
    *   *Fix:* **Sandwich Validation** (verify final claim against raw text).
3.  **Temporal Bias (Memory Poisoning):** An early false conclusion is reinforced by repetition.
    *   *Fix:* Give higher weight to "Synthesized Conclusions" over "Initial Hypotheses."

---

## 4. RLM vs Iron Core Interaction Policy

**Question:** Does RLM reading an Iron Core file (e.g., `01_PROTOCOLS/`) violate invariants?

**Policy:**
*   **READ Operations:** **ALLOWED.** RLM may recursively read/summarize `01_PROTOCOLS/`, `ADRs/`, and `founder_seed.json` to understand the constitution.
*   **WRITE Operations:** **FORBIDDEN.** RLM-generated summaries cannot *overwrite* Iron Core files without a standard Constitutional Amendment process (Protocol 110).
*   **Verification:** Any RLM summary of the Iron Core must be marked `[DERIVED ARTIFACT]` and never treated as the Constitution itself.
