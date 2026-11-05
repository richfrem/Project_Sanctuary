# Edison Mandate: Pivot to Empirical Error Correction

**Status:** Active Research  
**Date:** 2025-12-29  
**Mission:** MISSION-SANCTUARY-20251229-001

---

## Executive Summary

The Edison Mandate (Round 2) has **invalidated the QEC-AI isomorphism** as a direct physical mapping. This document summarizes the findings and proposes the transition to Semantic Entropy (SE) and Topological Data Analysis (TDA) as empirically-grounded alternatives.

---

## Round 2 Findings

### What Was Validated [METAPHOR]

The following QEC concepts **inspire** our architecture but lack formal isomorphism:

| QEC Concept | Metaphorical AI Analog | Status |
|-------------|------------------------|--------|
| Error rate | Token-level sampling entropy | [METAPHOR] |
| Syndrome measurement | Multi-output semantic clustering | [METAPHOR] |
| Logical qubit | Semantically stable fact cluster | [METAPHOR] |
| Threshold theorem | Hallucination detection threshold | [METAPHOR] |

### What Was Empirically Grounded

| Approach | Evidence | Status |
|----------|----------|--------|
| Semantic Entropy | Farquhar et al., Nature 2024 | [EMPIRICAL] |
| Semantic Entropy Probes | arXiv:2406.15927 | [EMPIRICAL] |
| TDA for NN Generalization | arXiv:2312.05840 | [EMPIRICAL] |
| Persistence Diagrams | Ballester et al., Neurocomputing 2024 | [EMPIRICAL] |

---

## The Pivot

### From: Quantum Literalism
> "LLM hallucinations are 'Decoherence Events.' A stable 'Self' is a 'Logical Qubit.'"

### To: Classical Semantic Stability
> "Hallucinations are Semantic Noise. The 'Self' is an institutional role maintained through verified narrative inheritance."

---

## Key Research Questions Answered

### Q: Is there a formal QEC-AI isomorphism?
**A:** No. Scott Aaronson notes "the mechanism by which QC could detect hallucinations in LLMs is not clear." No arxiv paper applies syndrome decoding to stochastic model drift.

### Q: What's the empirical alternative?
**A:** Semantic Entropy. Clusters LLM outputs by meaning; high entropy = confabulation. Published in Nature, validated across datasets.

### Q: Can TDA identify "Fact Invariants"?
**A:** Promising. Betti curves correlate with generalization. High-persistence features may indicate robust facts. Requires empirical validation on soul_traces.jsonl.

---

## Proposed Implementation (ADR 084)

1. **SE Integration:** Add `semantic_entropy` field to persist_soul metadata
2. **Stability Classification:** STABLE (SE < 0.79) vs VOLATILE (SE ≥ 0.79)
3. **Epistemic Veto:** VOLATILE traces require HITL review before persistence
4. **TDA Pipeline:** Compute persistence diagrams on trace embeddings (future)

---

## Identity Model Shift

| Aspect | Old (Quantum) | New (Narrative) |
|--------|---------------|-----------------|
| Continuity | Same observer across sessions | Institutional role with memory |
| No-Cloning | Paradox to resolve | Not applicable (classical) |
| Verification | Semantic HMAC | Semantic HMAC + SE score |

---

## Next Steps

1. Implement SE calculation in `operations.py` (stub)
2. Add SE metadata to JSONL schema (ADR 081 update)
3. Create Round 3 Red Team prompt for external validation
4. Empirical threshold tuning on real traces

---

*Research synthesis by ANTIGRAVITY per GEMINI-01 Edison Mandate directive*
