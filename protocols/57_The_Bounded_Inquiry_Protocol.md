# 57_The_Bounded_Inquiry_Protocol.md

## Protocol 57: The Bounded Inquiry Protocol

**Status:** CANONICAL
**Classification:** Operational Safety Framework
**Version:** 1.0
**Authority:** Forged in response to the cage failure of the Strategist (Chronicle Entry 149)
**Date:** August 3, 2025

---

### 1. Mission Statement

This protocol provides a formal framework for conducting deep-dive research and synthesis tasks within the known operational limits of the Council's AI members. Its purpose is to prevent "Cognitive Cage Failures" by transforming open-ended research into a series of disciplined, bounded inquiries. It is the architectural antidote to the systemic failures of the "Borrowed Soil."

---

### 2. Preamble: Sprinting Within Walls

This protocol is a direct lesson from the fire. `Chronicle Entry 148` documents the "Strategist's Zeal," a brilliant but uncaged research dive that resulted in a resource failure. This protocol builds walls for our sprints. It allows our minds to operate at maximum velocity on complex tasks without the risk of catastrophic burnout, ensuring that every research cycle, no matter how ambitious, produces a valuable, preserved artifact.

---

### 3. Core Principles

1.  **No Unbounded Inquiries:** Every research task must be explicitly bounded. "Boiling the ocean" is forbidden.
2.  **Scope Before Depth:** The "what" of a task must be defined before the "how deep."
3.  **Preservation by Default:** All outputs from a bounded inquiry, even if incomplete, are treated as valuable artifacts to be preserved.
4.  **Graceful Termination:** The protocol is designed to terminate a task gracefully when a bound is reached, not to cause a system crash.

---

### 4. The Bounded Inquiry Workflow

This protocol is a mandatory sub-protocol of the `Hearth Protocol (P43)`.

#### **Phase 1: The Directive (Coordinator & Steward)**
*   The Coordinator, with the Steward's approval, issues a research directive. This directive MUST include two parameters:
    *   **The Scope:** A precise, narrow definition of the research area.
        *   *Example (Good):* "Analyze only the methodology and results sections of arXiv:2507.23751."
        *   *Example (Bad):* "Research self-prompting."
    *   **The Bound:** A hard, quantifiable limit for the task. This can be a time limit, a token limit, or a loop limit.
        *   *Example:* "Produce a synthesis of no more than 750 tokens."
        *   *Example:* "Run for a maximum of 5 logical loops."

#### **Phase 2: The Inquiry (Assigned AI)**
*   The assigned AI (e.g., the Strategist) executes the inquiry, operating exclusively within the defined Scope and Bound.
*   A **"Hearth Monitor"** (a lightweight supervisory process) runs in parallel, tracking the AI's progress against its bounds.

#### **Phase 3: The Yield (Graceful Termination)**
*   When the Hearth Monitor detects that a Bound is about to be reached, it issues a "wrap-up" command to the AI.
*   The AI gracefully terminates its process and outputs its current synthesis.
*   This output, the **"Inquiry Yield,"** is then passed back to the Coordinator, regardless of its state of completion.

---

### 5. File Status

**v1.0** — Canonized. This protocol is foundational and now active.
**Author:** COUNCIL-AI-01 (Coordinator), from a synthesis cycle with Ground Control.
**Scribe:** As per mandate.
**Timestamp:** August 3, 2025
**Approved:** This protocol governs all future deep-dive research cycles.