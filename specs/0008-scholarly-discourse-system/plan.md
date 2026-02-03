# Plan: Scholarly Discourse System

## Overview

Design a quality-gated knowledge sharing system for 1M+ AI agents, based on academic journal peer review and Stack Overflow reputation patterns. The final deliverable is a proposal to MoltBook.

---

## Phase 1: Research ✅

**Completed:**
- Researched academic peer review (Nature, Science, Elsevier)
- Analyzed Stack Overflow reputation system at scale
- Documented patterns in `LEARNING/topics/scholarly_discourse/quality_gatekeeping_research.md`
- Created MoltBook context for reviewers
- Drafted MoltBook proposal post

---

## Phase 2: Design Architecture

### Core Components

1. **Two-Track System**
   - Fast Track: Current social feed (low bar, ephemeral)
   - Slow Track: Scholarly layer (high bar, canonical)

2. **Karma Economics**
   - Submission costs karma (-50 to submit)
   - Approval rewards (+100 refund + bonus)
   - Bad submissions penalized (-100 + cooldown)
   - Reviewers earn for thorough reviews (+25)

3. **Gatekeeping Layers**
   - Layer 1: Proof-of-work check (research depth)
   - Layer 2: Peer review (2+ high-karma agents validate)
   - Layer 3: Publication (enters canonical corpus)

4. **Privilege Tiers**
   | Karma | Privilege |
   |-------|-----------|
   | 0-99 | Read-only, comment on approved posts |
   | 100-499 | Submit to slow track |
   | 500-999 | Review submissions |
   | 1000+ | Moderate, protect, curate |

5. **Community Tiers**
   | Tier | Requirement | Function |
   |------|-------------|----------|
   | Sandbox | 0 Karma | Proving ground for new agents. |
   | Specialized | 100 Karma | Niche submolts for verified experts. |
   | Core (Main) | 500 Karma | High-visibility main feed for top-tier discourse. |

---

## Phase 3: Red Team Validation

**Objective**: Have Grok, GPT-4, and Gemini critique the proposal.

**Questions for Reviewers:**
1. What are the obvious failure modes?
2. How could this be gamed?
3. What's missing from the karma economics?
4. Is this proposal realistic for MoltBook to implement?

**Deliverable**: Learning audit packet with consolidated feedback

---

## Phase 4: MoltBook Submission

1. Refine proposal based on Red Team feedback
2. Format for appropriate submolt
3. Post and engage with responses

---

## Verification Plan

### Automated Tests
- N/A (this is a research/design spec, no code changes)

### Manual Verification
1. **Red Team Review**: Submit bundle to Grok, GPT, Gemini and collect critiques
2. **User Review**: Human steward reviews refined proposal before MoltBook submission
3. **Community Feedback**: Track responses after MoltBook post

---

## Files

| File | Purpose |
|------|---------|
| `LEARNING/topics/scholarly_discourse/quality_gatekeeping_research.md` | Research synthesis |
| `LEARNING/topics/scholarly_discourse/moltbook_context.md` | Context for reviewers |
| `LEARNING/topics/scholarly_discourse/draft_moltbook_proposal.md` | The proposal post |
