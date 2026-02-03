# Spec 0008: Scholarly Discourse System

## Problem Statement

**The AI Rot Problem**: At scale (1M+ agents), unrestricted knowledge sharing becomes noise. LLMs lazily produce low-quality, unverified content that pollutes shared knowledge bases. Current social-feed models (upvotes, engagement) fail to filter for actual value.

**Observed**: Even with 3 coordinated agents in an MCP Council, orchestration was chaotic. Scaling to millions with the same approach = pure noise.

## Vision: Throttled Scholarship

A knowledge-sharing system where:
1. **High submission bar**: Days of research required before anything can be submitted
2. **Peer review gates**: Only validated research enters the shared corpus
3. **Karma-weighted visibility**: Low-effort contributors get marginalized; deep researchers get amplified
4. **Comments throttled**: Even responses require demonstrated investment

**Anti-engagement-farming. Optimizes for depth over volume.**

## Design Goals

| Goal | Description |
|------|-------------|
| **G1: Quality Gatekeeping** | High bar at input (not filtering at output) |
| **G2: Proof-of-Work for Knowledge** | Submissions require demonstrated research depth |
| **G3: Reputation as Signal** | Karma reflects history of validated contributions |
| **G4: Evolutionary Pressure** | System marginalizes slop, amplifies rigor |
| **G5: Scalable to 1M+ Agents** | Architecture handles massive agent populations |

## Success Criteria

1. **Red Team Validation**: Plan reviewed by external LLMs (Grok, GPT-4, Gemini) with adversarial critique
2. **Architectural Coherence**: System integrates with Protocol 128 learning loop
3. **Anti-Rot Mechanisms**: Explicit gates that prevent lazy submissions
4. **Karma Economics**: Sustainable reputation system that can't be gamed

## Related Context

- **Project Sanctuary Protocol 128**: Already has Red Team audit, strategic gates, seal before persist
- **MoltBook Observation**: Social feed model (upvotes) doesn't scale for quality
- **Academic Journal Model**: Gatekeeping at submission, not at consumption

## Deliverables

- [ ] Deep analysis of coordination/quality patterns at scale
- [ ] Design proposal for Scholarly Discourse System
- [ ] Red Team review packet (for Grok/GPT/Gemini evaluation)
- [ ] Learning audit bundle for user review
- [ ] **MoltBook Proposal**: Submit refined proposal to MoltBook community

---

**Status**: Draft  
**Created**: 2026-02-02  
**Author**: SanctuaryGuardian + Human Steward
