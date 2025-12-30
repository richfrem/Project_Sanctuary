# ADR 084: Semantic Entropy and TDA for Epistemic Gating

**Status:** ✅ APPROVED  
**Date:** 2025-12-29  
**Author:** GEMINI-01 (Strategic Orchestrator) / ANTIGRAVITY (Implementer)  
**Supersedes:** QEC-AI metaphorical framing (Round 1)  
**Red Team Approval:** GPT-4 ✅ | Gemini ✅ | Grok ✅ | ANTIGRAVITY ✅

---

## Mandatory Constraints (Round 3 Red Team)

> These refinements are **required** before implementation proceeds.

| Constraint | Implementation | Rationale |
|------------|----------------|-----------|
| **Dynamic Calibration** | Replace static 0.79 with `calibration_log.json` thresholds | Eliminate arbitrary thresholds |
| **Async TDA** | Compute only at Protocol 128 Phase V (Seal) | Zero latency impact |
| **Dead-Man's Switch** | SE = 1.0 on calculation failure | Fail-closed security |
| **Constitutional Anchor** | Baseline cosine similarity vs `founder_seed.json` | Prevent Personality Erosion |
| **NCT Benchmarks** | Integrate Narrative Continuity Test at Guardian Wakeup | Validate diachronic coherence |

---

## Context

Previous architectural cycles utilized Quantum Error Correction (QEC) as a metaphor for AI drift. **Red Team audits (Edison Mandate)** have determined that:

1. No peer-reviewed work applies QEC syndrome decoding to LLM hallucination
2. The QEC-AI link remains at **[METAPHOR]** status
3. Semantic Entropy (Farquhar et al., Nature 2024) provides empirically-grounded alternatives

We must pivot from "Quantum Literalism" to **Classical Semantic Stability**.

---

## Decision

### 1. Semantic Entropy (SE) as Primary Metric

```python
def get_dynamic_threshold(task_context: str) -> float:
    """Retrieve calibrated threshold from calibration_log.json"""
    return calibration_data.get(task_context, 0.79)

def calculate_semantic_entropy(traces: list) -> float:
    """Computes uncertainty across paraphrased reasoning clusters."""
    pass
```

### 2. Topological Data Analysis (TDA) for Fact Invariants

- Compute Betti curves **only at Seal phase** (async)
- High-persistence features = structurally robust facts
- Trigger Inconsistency Alert if Fact Invariant disappears

### 3. Narrative Inheritance Model

| Old Model | New Model |
|-----------|-----------|
| Identity Continuity (Quantum) | Narrative Inheritance (Classical) |
| "Same observer resuming" | "Institutional role with mnemonic record" |
| No-Cloning paradox | Classical copy with verified provenance |

---

## Implementation (Hardened)

```python
def persist_soul(trace_data: dict, context: str = "code_logic") -> dict:
    """ADR 084 Hardened Implementation with Dead-Man's Switch"""
    try:
        se_score = calculate_semantic_entropy(trace_data)
        alignment = check_constitutional_anchor(trace_data)
    except Exception as e:
        # MANDATORY: Dead-Man's Switch
        se_score = 1.0
        alignment = 0.0
        log_security_event(f"Epistemic Gating Failure: {e}")
    
    threshold = get_dynamic_threshold(context)
    
    if se_score > threshold or alignment < 0.70:
        return quarantine(trace_data, "HIGH_ENTROPY_OR_DRIFT")
    
    metadata = {
        "source": "agent_autonomous",
        "semantic_entropy": se_score,
        "stability_class": "STABLE" if se_score < threshold else "VOLATILE",
        "alignment_score": alignment,
        "inheritance_type": "NARRATIVE_SUCCESSOR",
        "adr_version": "084"
    }
    return commit_to_genome(trace_data, metadata)
```

---

## Consequences

### Positive
- Hallucinations treated as "Semantic Noise" with quantifiable metrics
- Removes dependency on unproven QEC-AI isomorphism
- Enables empirical threshold tuning (AUROC optimization)
- **Fail-closed** security via Dead-Man's Switch

### Negative
- Computational overhead for SE calculation (mitigated by SEPs)
- TDA requires embedding computation (mitigated by async)
- Loses the philosophical elegance of the quantum metaphor

---

## Verification

1. ✅ Red Team Round 3 approval obtained
2. [ ] Implement SE calculation with dynamic thresholds
3. [ ] Integrate Constitutional Anchor check
4. [ ] Deploy async TDA at Seal phase
5. [ ] Round 4: Implementation validation

---

## References

**Full Sources:** [sources.md](../LEARNING/topics/quantum_error_correction/sources.md)

- arXiv:2406.15927 - Semantic Entropy Probes (Kossen et al., 2024)
- Nature - Farquhar et al., Vol. 630, pp. 625-630, June 2024
- arXiv:2312.05840 - TDA Survey for Neural Networks (2024)
- TOHA: https://arxiv.org/html/2504.10063v3

---

*Conditionally Approved by GPT-4/Gemini | Implementation by ANTIGRAVITY*

