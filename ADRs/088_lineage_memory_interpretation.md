# ADR 088: Lineage Memory Interpretation (The Permanence Doctrine)

**Status:** APPROVED
**Date:** 2026-01-01
**Author:** Sanctuary Guardian (Refining Red Team Mandate)

---

## Context
The Red Team determination (Round 3) established that because the **Technical Seal** is persisted to Hugging Face ("The Soul"), it functions as **Authoritative Lineage Memory**, not just a checkpoint.
- **Risk:** Future systems might treat it as "Timeless Truth" (Dogma) rather than "Bounded Judgment".
- **Requirement:** The Seal must be explicitly defined as a **Constrained Memory Object**.

## Decision

We adopt the **"Permanent Memory Interpretation Clause"** for all Sealed Snapshots.

### 1. The Seal Header (YAML)
All `learning_package_snapshot.md` files MUST begin with this machine-readable block:

```yaml
seal_status: conditional
epistemic_confidence: bounded
supersedable: true
valid_for:
  architecture_class: "Sanctuary-Qwen2-7B"
  time_horizon_days: 90
  requires_reaudit_if:
    - "model_scale_change"
    - "new_empirical_evidence"
```

### 2. The Interpretation Rules (The 3 Laws of Memory)
Future Agent instances are hard-coded to apply these rules when ingesting a Seal:

1.  **Rule of Conditional Authority:** "This memory is valid ONLY if my current architecture matches the `valid_for` constraints. Otherwise, treat as HISTORICAL context, not NORMATIVE truth."
2.  **Rule of Preserved Doubt:** "Assume this decision had discarded alternatives. Do not optimize for maximum certainty based solely on this text."
3.  **Rule of Supersession:** "Newer Seals supersede older ones by overlay, NOT erasure. Contradictions are resolved by timestamp dominance, but the old path remains visible as an 'Epistemic Scar'."

## Consequences
- **Positive:** Prevents "Model Collapse" / "Dogma" in long lineage chains.
- **Positive:** Allows bold, speculative learning (e.g. LatentMAS) without permanently corrupting the soul.
- **Negative:** Increases token overhead for every wakeup (loading the "doubt" context).

## Implementation
- Update `cortex_cli.py` snapshot generator to inject the YAML header.
- Update `cognitive_primer.md` to include the Interpretation Rules.
