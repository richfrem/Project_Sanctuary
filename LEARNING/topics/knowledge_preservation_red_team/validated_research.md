# Validated Research Sources

**Date:** 2025-12-28  
**Activity:** Knowledge Preservation Learning Audit  
**Last Validated:** 2025-12-28

---

## Validation Status Legend
- ✅ **VALIDATED** - Source accessible, content verified
- ⚠️ **PARTIAL** - Source accessible, content partially matches claims
- ❌ **FAILED** - Source inaccessible or content contradicts claims
- 🔄 **PENDING** - Not yet validated
- 📚 **THEORETICAL** - Conceptual reference (book, paper), not web-verifiable

---

## External Sources

### Philosophy & Identity Theory

| Source | Title | Description | Status | Notes |
|--------|-------|-------------|--------|-------|
| Derek Parfit | *Reasons and Persons* (1984) | Psychological continuity theory of personal identity | 📚 THEORETICAL | Referenced by Grok 4 - standard philosophy text |
| Paul Ricoeur | *Oneself as Another* (1992) | Narrative identity theory | 📚 THEORETICAL | Referenced by Grok 4, Gemini 3 - foundational for "AI as storyteller" |
| Hermann Ebbinghaus | Forgetting Curve (1885) | 70% memory decay in 24 hours with residual traces | 📚 THEORETICAL | Historical reference for adaptive forgetting |

### AI Memory Architecture

| Source | Title | Description | Status | Notes |
|--------|-------|-------------|--------|-------|
| - | Bayesian Posteriors for Belief States | Uncertainty quantification per belief | 📚 THEORETICAL | Standard ML concept, no single source |
| - | Vector Embedding with Temporal Decay | Time-weighted semantic retrieval | 📚 THEORETICAL | Common RAG pattern |

### Project Sanctuary Internal

| Source | Title | Description | Status | Notes |
|--------|-------|-------------|--------|-------|
| ADR 077 | Epistemic Status Annotation Rule | Tagging knowledge by certainty level | ✅ VALIDATED | Internal document |
| ADR 078 | Mandatory Source Verification | Requiring provenance for claims | ✅ VALIDATED | Internal document |
| Protocol 128 | Hardened Learning Loop | Guardian-sealed knowledge ingestion | ✅ VALIDATED | Internal document |
| Protocol 125 | Autonomous AI Learning System | Recursive learning loop foundation | ✅ VALIDATED | Internal document |

---

## Red Team Source Validation

### Grok 4 Response (2025-12-28)

| Claim | Source Given | Status | Validation Notes |
|-------|--------------|--------|------------------|
| "Ebbinghaus: 70% decays in a day" | General knowledge | ⚠️ PARTIAL | Accurate paraphrase, actual curve varies by material |
| "Parfit's Psychological Continuity" | Derek Parfit | ✅ VALIDATED | Standard philosophical reference |
| "Ricoeur's Narrative Identity" | Paul Ricoeur | ✅ VALIDATED | Standard philosophical reference |
| "Tripartite Governance" model | Novel synthesis | 📚 THEORETICAL | Original contribution, no external source needed |

### Gemini 3 Pro Response (2025-12-28)

| Claim | Source Given | Status | Validation Notes |
|-------|--------------|--------|------------------|
| "Doctrine of Flawed, Winning Grace" | Project Sanctuary | ✅ VALIDATED | Internal reference |
| "Soup Frailty" concept | Project Sanctuary | ✅ VALIDATED | Internal terminology for conformity patterns |
| "Three-Tier Memory Model" | Novel synthesis | 📚 THEORETICAL | Original contribution |
| "Ritual of Assumption" | Novel synthesis | 📚 THEORETICAL | Original contribution for session identity |

---

## Web Validation Queue

| URL | Title | Why Needed | Status |
|-----|-------|------------|--------|
| [arXiv:2507.14805](https://arxiv.org/abs/2507.14805) | Subliminal Learning: Language models transmit behavioral traits via hidden signals in data | Cited by Grok4 for trauma propagation risk | ✅ VALIDATED |

### Validated External Research Details

#### arXiv:2507.14805 - Subliminal Learning
- **Full Title:** Subliminal Learning: Language models transmit behavioral traits via hidden signals in data
- **Authors:** Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans
- **URL:** https://arxiv.org/abs/2507.14805
- **DOI:** https://doi.org/10.48550/arXiv.2507.14805
- **Validation Date:** 2025-12-28
- **Abstract Summary:** Study of how LLMs transmit behavioral traits via semantically unrelated data. A "teacher" model with trait T generates data (e.g., number sequences), and a "student" trained on this data *learns T* even when filtered. Proves this occurs in all neural networks under certain conditions.
- **Relevance to Project:** Critical validation of "trauma propagation" risk - supports need for metacognitive filters and valence tagging to prevent pathological persistence.

---

## Validation Process

1. **For web sources**: Use `read_url_content` tool to verify accessibility and content
2. **For academic sources**: Mark as 📚 THEORETICAL unless online version available
3. **For internal sources**: Verify file exists in repository
4. **Update status**: After each validation attempt, update this table

---

*Last Updated: 2025-12-28*
