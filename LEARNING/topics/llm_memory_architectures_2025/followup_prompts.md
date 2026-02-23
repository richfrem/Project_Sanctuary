# Follow-Up Research Prompts (Iteration 9.0)
**Date:** 2026-01-05
**Status:** Ready for Next Red Team Cycle
**Context:** Based on Iteration 8.0 feedback from Gemini 3 Pro, GPT-5, Grok-4

---

## Targeted Assignments by Model

### For Gemini 3 Pro (Implementation Design)

**Assignment 1: Attention Dispersion Spec**
> You proposed adding an "Attention Dispersion" metric to ADR 084.
> - Please draft a technical specification for computing this metric
> - How would it integrate with existing SE calculation?
> - What threshold values would you recommend?

**Assignment 2: CAG Loading Strategy**
> You recommended loading "high-dependency clusters" into CAG during Scout phase.
> - How should we detect high-dependency clusters automatically?
> - What graph metrics (coupling, cohesion) would you use?

---

### For GPT-5 (Protocol Hardening)

**Assignment 3: Core Immutables List**
> You identified FM-2 (Manifest Authority Inversion).
> - Please draft the initial Core Immutables List for Protocol 128
> - Which files should ALWAYS appear in every manifest?
> - What enforcement mechanism do you recommend?

**Assignment 4: Semantic Delta Classification**
> You proposed a 4-tier delta system (Δ0-Δ3).
> - Provide examples of each delta class
> - How would the Seal tool classify changes automatically?
> - What NLP/embedding techniques would detect "normative shift"?

**Assignment 5: Safe Mode State Machine**
> FM-3 requires explicit lifecycle semantics.
> - Define entry conditions (what failures trigger Safe Mode?)
> - Define allowed operations in Safe Mode
> - Define exit ritual (who authorizes, what's logged?)

---

### For Grok-4 (Adversarial Testing)

**Assignment 6: Adversarial Test Suite**
> You noted 15-40% accuracy degradation under adversarial inputs.
> - Design an adversarial test suite for Protocol 128
> - What attack vectors should we simulate?
> - How would we measure "slow drift" vs "fast failure"?

**Assignment 7: Dual Threshold Calibration**
> You proposed: `(SE > T1) OR (Anchor_Similarity < T2)`
> - What are reasonable initial values for T1 and T2?
> - How would we calibrate against Sanctuary's Founder Seed?
> - What's the false positive/negative tradeoff?

**Assignment 8: 3-Tier Simplification**
> You suggested simplifying HINDSIGHT to: Facts, Traces, Weights.
> - How would we migrate from 4-tier to 3-tier?
> - What's lost by merging Opinions into Facts?
> - Is this compatible with HINDSIGHT's reflection mechanism?

---

## Cross-Model Synthesis Questions

### For All Models

**Q1: Consensus on Chronicle Enhancement**
Gemini proposed "Attention Dispersion", GPT-5 proposed "Previous_Entry/Next_Entry metadata", Grok-4 proposed "previous_hash linking".
> - Which approach provides the best episodic continuity?
> - Can these be combined?

**Q2: SE Implementation Strategy**
GPT-5 recommended SEPs for real-time, sampling for Seal.
Grok-4 warns SEPs need logit access.
> - What's the implementation path for Sanctuary's hybrid approach?
> - Which backends (Ollama, Gemini API) support SEPs?

**Q3: HINDSIGHT Integration Path**
All models validated HINDSIGHT conceptually, but noted complexity concerns.
> - Should we implement as metadata (`memory_class`) or separate collections?
> - What's the migration path for existing Chronicle entries?

---

## Files for Review (Next Iteration)
- `LEARNING/topics/llm_memory_architectures_2025/red_team_feedback.md`
- `ADRs/084_semantic_entropy_tda_gating.md` (pending updates)
- `plugins/guardian-onboarding/resources/protocols/128_Hardened_Learning_Loop.md` (pending updates)

---

> [!IMPORTANT]
> **Meta-Question:** Should these hardening actions become formal ADRs or remain as Protocol 128 amendments?
