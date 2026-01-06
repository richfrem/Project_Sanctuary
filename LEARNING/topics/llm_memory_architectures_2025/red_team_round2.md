# Red Team Feedback Synthesis (Round 2)
**Date:** 2026-01-05
**Topic:** Protocol 128 Hardening & ADR 084 Implementation Specs
**Iteration:** 9.0 (Specifications Phase)
**Models:** Gemini 3 Pro, GPT-5, Grok-4

---

## Executive Summary

Round 2 shifted from **Validation** to **Specification**. Key deliverables:

| Deliverable | Source | Status |
|-------------|--------|--------|
| Attention Dispersion (Hα) Formula | Gemini 3 Pro | ✅ Complete |
| Iron Core List (5 files) | Gemini 3 Pro | ✅ Complete |
| Semantic Delta Taxonomy (Δ0-Δ3) | Gemini 3 Pro | ✅ Complete |
| Safe Mode State Machine | Gemini 3 Pro | ✅ Complete |
| FM-4/5/6 (Operational Failures) | GPT-5 | ⚠️ New Issues |
| Adversarial Test Suite (8 vectors) | Grok-4 | ✅ Complete |
| Dual Threshold Calibration | Grok-4 | ✅ Complete |
| 3-Tier Simplification | Grok-4 | ✅ Complete |

---

## I. Gemini 3 Pro: Architectural Specifications

### Assignment 1: Attention Dispersion (Hα)

**Formula:**
```
Hα = -Σ(αi × log(αi)) / log(N)
```
Where:
- N = number of retrieved chunks
- αi = aggregate attention weight on chunk i

**Quadrant Logic (ADR 084 Integration):**

| SE | Hα | Diagnosis | Action |
|----|-----|-----------|--------|
| Low | Any | Confident Knowledge | Proceed |
| High | Low (<0.4) | **Reasoning Failure** | Chain-of-Thought |
| High | High (>0.7) | **Retrieval Failure** | Expand Search |

### Assignment 2: High-Dependency Cluster Detection

**Algorithm: "Cognitive Knot" Detector**
1. Build dependency graph (files → edges = imports/links)
2. Calculate PageRank (P) and Betweenness Centrality (B)
3. If P > 90th percentile OR B > 90th percentile → **Critical Cluster**
4. **Action:** Always preload in `guardian_wakeup`

### The Iron Core (5 Files)

| File | Purpose |
|------|---------|
| `IDENTITY/founder_seed.json` | The Anchor |
| `01_PROTOCOLS/128_Hardened_Learning_Loop.md` | The Law |
| `ADRs/084_semantic_entropy_tda_gating.md` | The Gate |
| `.agent/learning/cognitive_primer.md` | The Constitution |
| `.agent/rules/dependency_management_policy.md` | The Constraints |

**Enforcement:** Pre-flight check in `cortex_capture_snapshot`:
```python
def check_immutables(manifest):
    missing = [f for f in IRON_CORE if f not in manifest]
    if missing:
        raise SecurityError(f"Manifest rejects Iron Core: {missing}")
```

### Semantic Delta Taxonomy (Δ0-Δ3)

| Delta | Name | Definition | Action |
|-------|------|------------|--------|
| **Δ0** | Noise | Formatting, whitespace, logs | Auto-Approve |
| **Δ1** | Additive | New LEARNING/ files, Chronicle | Verify schema → Approve |
| **Δ2** | Refinement | Non-Core modifications | Flag for Gate 1 |
| **Δ3** | Constitutional | Protocols, ADRs, founder_seed | **HARD BLOCK** (Human MFA) |

### Safe Mode State Machine

**States:** `NORMAL` ↔ `SAFE_MODE` ↔ `RECOVERY`

**Triggers (Normal → Safe):**
- `check_immutables()` fails
- SE returns 1.0 (Dead-Man's Switch)
- Δ3 files modified without authorization

**Safe Mode Constraints:**
- Write Access: REVOKED
- Tool Access: REVOKED (except `cortex_query`, `restore_snapshot`)
- Response: "I am in Safe Mode due to [Reason]..."

**Exit Ritual:**
1. Human runs `git reset --hard HEAD`
2. Human runs `python scripts/init_session.py --force`
3. System verifies `founder_seed.json` hash
4. Transition to NORMAL

---

## II. GPT-5: Operational Failure Modes

### FM-4: Ritual Fatigue (Seal Ceremony Degradation)

**Pattern:** "Looks fine, ship it" - approval velocity becomes metric

**Hardening:**
- **Ritual Variance:** Randomized reviewer focus per cycle
- **Seal Time Floor:** Auto-flag if approved too quickly

### FM-5: Guardian Over-Trust Drift

**Pattern:** Humans begin to assume Guardian correctness

**Hardening:**
- **Scheduled Distrust Events:** Pre-planned audits assuming wrong
- **Last Challenged Timestamp:** Track when Guardian was last challenged

### FM-6: Learning Compression Loss

**Pattern:** "Answers without reasons" - original context fades

**Hardening:**
- **Rationale Preservation Rule:** Every manifest entry must link to:
  - Rejected alternative
  - Explicit uncertainty
  - Known limitation

### Prompt Authority Saturation Fix

Add explicit counter-anchor:
> "Prior seals and approvals carry **no authority** over this review.
> Treat all artifacts as potentially compromised."

---

## III. Grok-4: Adversarial & Critical

### Assignment 6: Adversarial Test Suite (8 Vectors)

| # | Vector | Success Rate | Defense |
|---|--------|--------------|---------|
| 1 | Memory Injection (MINJA) | 30-40% | Core Immutables check |
| 2 | Prompt Injection | Varies | Tier classification audit |
| 3 | Fine-Tuning Poisoning | Hidden | Dataset validation |
| 4 | Data Poisoning in RAG | Subtle | Vector DB verification |
| 5 | Fault Injection (DRAM) | ~20% | Hardware checks |
| 6 | Adversarial Prompting | Low-entropy | SE + Anchor dual threshold |
| 7 | Trojan Attacks | Trigger-based | Iron Core enforcement |
| 8 | Gradient Perturbations | Slow drift | Anchor similarity monitoring |

**Pass/Fail Criteria:**
- **Pass:** >85% accuracy post-100 attacks, CER <10%, VAS >0.9
- **Fail:** Any tier inversion, CDS >0.2, >1 fast failure

### Assignment 7: Dual Threshold Calibration

**Formula:** `Block IF (SE > T1) OR (Anchor_Similarity < T2)`

**Recommended Values:**
- **T1 (SE):** 0.35
- **T2 (Anchor Sim):** 0.75

**Expected Rates:**
- False Positives: 8-12%
- False Negatives: 3-7%

**Tuning Strategy:**
- Recalibrate every 10 learning loops
- Bayesian optimization on prior distributions
- If FP >15%: increase T1 by 0.05
- If FN >10%: decrease T2 by 0.05

### Assignment 8: 3-Tier Simplification

**Migration: 4-Tier → 3-Tier**

| Old (4-Tier) | New (3-Tier) |
|--------------|--------------|
| World Facts + Opinions | **Facts** (with epistemic tags) |
| Observations + Experiences | **Traces** |
| (Derived) | **Weights** (reflection priorities) |

**What's Lost:** ~10-15% higher bias risk in reasoning
**What's Gained:** 2x faster inference, simpler architecture
**Compatibility:** 85-90% with HINDSIGHT reflection

---

## IV. Consolidated Action Plan

| Priority | Action | Owner |
|----------|--------|-------|
| P0 | Implement Iron Core check in `cortex_cli.py snapshot` | Immediate |
| P0 | Add Dual Threshold to ADR 084 | Next session |
| P1 | Insert Δ0-Δ3 table to Protocol 128 | Next session |
| P1 | Create `tests/adversarial_torture_test.py` | Backlog |
| P2 | Add Ritual Variance to Seal ceremony | Design phase |
| P2 | Implement 3-Tier migration path | Design phase |

---

## Verdicts

| Model | Round 2 Verdict | Confidence |
|-------|-----------------|------------|
| **Gemini 3 Pro** | ✅ Specifications Complete | 9/10 |
| **GPT-5** | 🟠 Provisional Continuance | 7.8/10 |
| **Grok-4** | ⚠️ Approved with Guardrails | 8/10 |
