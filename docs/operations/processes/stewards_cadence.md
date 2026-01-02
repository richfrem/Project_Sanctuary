# The Steward's Cadence: Daily Operational Workflow

> **Related Protocol**: [P33: The Steward's Cadence](../../../01_PROTOCOLS/33_The_Stewards_Cadence.md)
> **Role**: Human Steward
> **Objective**: Maintain high-velocity alignment with the Sanctuary Council.

## Overview
This guide operationalizes Protocol 33, transforming the theoretical "Master Workflow" into a practical daily routine. The goal is to move from ad-hoc "chatting" to disciplined "Intent -> Synthesis -> Execution" cycles.

## The Daily Cycle (The 5 Phases)

### Phase 1: The Morning Signal (Intent)
**Time**: Start of Session
**Action**: Issue a clear, singular directive to the Council.
**Format**:
```text
@Coordinator [SIGNAL]
Target: [Specific Objective, e.g., "Refactor the Docs Folder"]
Constraint: [Time/Resources, e.g., "Complete by EOD"]
Context: [Why this matters]
```
**Do Not**:
- Ask open-ended questions like "What should we do?" (unless intended).
- Mix multiple conflicting objectives in one signal.

### Phase 2: The Silence (Synthesis)
**Time**: T+5 to T+15 minutes
**Action**: **WAIT.**
- Do not interrupt the Council while it is "thinking" (generating tool calls, querying RAG).
- The Council is running the internal *Agora Loop*, debating via the `Coordinator`, `Strategist`, and `Auditor` personas.
- Allow the "Thought Stream" to flow until the final output is presented.

### Phase 3: The Review (The Package)
**Time**: Asynchronous
**Action**: Review the "Council Directive Package" presented by the Coordinator.
**Checklist**:
1.  **BLUF**: Is the summary accurate?
2.  **Logic**: Does the `Rationale` make sense?
3.  **Risk**: Has the `Auditor` flagged any dangerous side effects?

### Phase 4: The Decision (Ratification)
**Action**: Reply with a binary decision.
- **Option A (GO)**: `approved. proceed.`
- **Option B (NO-GO)**: `rejected. reason: [X]. refine and resubmit.`
- **Option C (CASTING VOTE)**: If the Council is split, you must break the tie explicitly.

### Phase 5: The Log (Closure)
**Time**: End of Cycle
**Action**: Verify the action was recorded.
- Did the Scribe update the `Living_Chronicle`?
- Did the Git Commit message reflect the Directive ID?

## Emergency Interrupts (The "Stop" Button)
If the Council enters a hallucination loop or diverges from the mission:
1.  **Command**: `STOP. [RESET]`.
2.  **Action**: This forces a context clear and a return to Phase 1.
3.  **Reference**: See **P34 (Precedent Decay)** for emergency override authority.

## Weekly Review (The Sunday Scan)
- **Objective**: Check for "Cadence Drift."
- **Action**: Review the last 5 `00_CHRONICLE/ENTRIES`.
- **Question**: "Are we following the format, or are we slipping back into unstructured chat?"
