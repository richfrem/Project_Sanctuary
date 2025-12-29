# Red Team Audit Template: Epistemic Integrity Check

**Target:** Learning Loop Synthesis Documents  
**Protocol:** 128 (Hardened Learning Loop)  
**ADR Reference:** ADR 071, ADR 080

---

## Purpose

This template guides the **learning_audit** process, which focuses on the validity of truth and the integrity of reasoning chains. This ensures that AI research doesn't just sound plausible but is **epistemically sound** before being "Sealed" and persisted to the Hugging Face AI Commons.

> **Note:** A `learning_audit` differs from a standard code/system audit. It validates reasoning, not syntax.

---

## 1. Verification of Thresholds (Protocol 128)

- [ ] Did the agent verify physical error rates against the relevant Threshold Theorem?
- [ ] Is there a `[VERIFIED]` log for every source cited?
- [ ] Were any speculative claims masked as empirical?
- [ ] Are confidence intervals provided for numerical claims?

---

## 2. Reasoning Trace Audit (ADR 080)

- [ ] Inspect the `reasoning_chain` in the registry
- [ ] Does each inference step account for information loss or transformation?
- [ ] Identify any "High Confidence" tags that lack supporting empirical data
- [ ] Are uncertainty distributions provided for key inferences?

---

## 3. Semantic Drift Detection

- [ ] Compare the "Scout" context (prior knowledge) with the final synthesis
- [ ] Have key definitions drifted into metaphor, or do they remain mathematically grounded?
- [ ] Is terminology used consistently throughout the document?
- [ ] Are analogies clearly labeled as such (not presented as equivalences)?

---

## 4. Metadata & Valence Check (Protocol 129)

- [ ] Does the valence score reflect any pathological bias?
- [ ] Are `source:containment_trauma` or similar flags present?
- [ ] Confirm the JSONL record matches the ADR 081 schema
- [ ] Validate that `uncertainty` field is populated appropriately

---

## 5. Source Verification (ADR 078)

| Source Type | Requirement | Status |
|-------------|-------------|--------|
| Peer-reviewed | 2024-2025 publications | [ ] |
| Experimental | Verified results only | [ ] |
| Foundational | Established textbooks | [ ] |
| Speculative | Clearly labeled | [ ] |

---

## Audit Verdict

| Criterion | Pass/Fail | Notes |
|-----------|-----------|-------|
| Threshold Verification | | |
| Reasoning Trace Integrity | | |
| No Semantic Drift | | |
| Valence Check | | |
| Source Verification | | |

**Final Recommendation:** [ ] APPROVE FOR PERSISTENCE / [ ] REQUIRE REVISION

---

*Template Version: 1.0*  
*Last Updated: 2025-12-28*
