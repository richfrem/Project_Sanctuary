# Learning Audit Prompt: DRQ Application to Sanctuary
**Current Topic:** Digital Red Queen - Recursive Self-Improvement
**Iteration:** 1.0 (Initial Red Team)
**Date:** 2026-01-11
**Epistemic Status:** [PROPOSED - AWAITING RED TEAM VALIDATION]

---

> [!NOTE]
> For foundational project context, see `learning_audit_core_prompt.md` (included in this packet).

---

## 📋 Topic: Applying DRQ + Related Principles to Project Sanctuary

### Research Summary

Deep analysis of **recursive self-improvement** patterns from multiple sources:

| System | Key Pattern | Sanctuary Application |
|:-------|:------------|:---------------------|
| **DRQ (Sakana AI)** | LLM evolution + cumulative adversaries | Red Team accumulates ALL edge cases |
| **Map-Elites** | Quality-Diversity archive | Track outputs by (depth, scope) grid |
| **AlphaGo Zero** | Tabula rasa + self-play | Enable emergent insights beyond sources |
| **FunSearch (DeepMind)** | LLM + automated evaluator | Formalize pre-Red-Team validation |
| **AI Scientist (Sakana)** | LLM reviewer loop | Iterative refinement until quality threshold |

### Files Created

| Artifact | Purpose |
|:---------|:--------|
| `README.md` | Topic overview |
| `sources.md` | ADR 078 verified sources |
| `notes/drq_paper_analysis.md` | Paper deep dive |
| `notes/sanctuary_evolution_proposal.md` | High-level proposal |
| `notes/learning_loop_technical_synthesis.md` | Code-level DRQ analysis |
| `notes/related_work_research.md` | **NEW** Map-Elites, AlphaGo, FunSearch research |

> [!IMPORTANT]
> **For Deep Dive:** Clone and analyze the DRQ repo directly:
> ```bash
> git clone https://github.com/SakanaAI/drq.git
> ```
> Key files: `src/drq.py` (Map-Elites + evolution loop), `src/prompts/` (minimal prompts)

---

## 🎭 Red Team Focus (Iteration 1.0)

### Primary Questions

1. **PATTERN: Map-Elites Behavioral Archive**
   - Should we track learning outputs by (depth, scope) axes?
   - Key metrics: Coverage (fraction filled), QD-Score (total fitness)
   - How to objectively measure "depth" and "scope"?

2. **PATTERN: Automated Pre-Evaluation**
   - FunSearch pairs LLM with automated evaluator to catch hallucinations
   - Should we formalize internal evaluation BEFORE Red Team?
   - Reduce human burden, catch obvious issues early

3. **PATTERN: Cumulative Edge Case History**
   - DRQ evaluates against ALL previous champions, not just latest
   - Each Red Team review should accumulate edge cases
   - New outputs must pass ALL accumulated tests

4. **PATTERN: Prompt Simplification**
   - DRQ mutation prompt: ~300 chars
   - `sanctuary-guardian-prompt.md`: ~30KB
   - Split into: action prompts (~300 chars) + domain context (~15KB)?

5. **META: Emergent Discovery vs Knowledge Reproduction**
   - AlphaGo Zero discovered novel strategies beyond human knowledge
   - Should learning outputs aim for emergence or just reproduction?
   - Balance: Risk of hallucination vs value of novel insights

---

## 📁 Files in This Packet

**Total:** 14 files (8 core + 6 topic-specific)

### Core Context (8 files)
- `README.md`, `IDENTITY/founder_seed.json` - Identity
- `cognitive_primer.md`, `guardian_boot_contract.md` - Boot sequence
- `01_PROTOCOLS/128_Hardened_Learning_Loop.md` - Protocol
- `learning_audit_core_prompt.md`, `learning_audit_prompts.md` - Audit context
- `cognitive_continuity_policy.md` - Rules

**Total:** 16 files

### Core Context (10 files)
- `README.md`, `IDENTITY/founder_seed.json`
- `cognitive_primer.md`, `guardian_boot_contract.md`
- `01_PROTOCOLS/128_Hardened_Learning_Loop.md` (**UPDATED v4.0 Proposal**)
- `01_PROTOCOLS/131_Evolutionary_Self_Improvement.md` (**NEW**)
- `learning_audit_core_prompt.md`, `learning_audit_prompts.md`
- `cognitive_continuity_policy.md`
- `docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd` (**UPDATED**)

### Topic-Specific (8 files)
- `drq_recursive_self_improvement/README.md`
- `sources.md`
- `notes/drq_paper_analysis.md`
- `notes/sanctuary_evolution_proposal.md`
- `notes/learning_loop_technical_synthesis.md`
- `notes/related_work_research.md`
- `notes/plain_language_summary.md` (**NEW**)
- `docs/architecture_diagrams/workflows/drq_evolution_loop.mmd` (**NEW**)

---

> [!IMPORTANT]
> **Goal:** Validate the *Protocol* for Evolutionary Self-Improvement before implementing the code triggers.
