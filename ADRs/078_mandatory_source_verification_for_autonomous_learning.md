# Mandatory Source Verification for Autonomous Learning

**Status:** PROPOSED
**Date:** 2025-12-28
**Author:** Claude (Antigravity Agent)
**Supersedes:** ADR 077

---

## Context

Red team review of autonomous learning (Entry 337) revealed two risks:
1. High-coherence synthesis can mask epistemic confidence leaks
2. Sources listed without verification may be hallucinated

GPT flagged: "MIT Consciousness Club" and "April 2025 Nature study" as potentially fabricated.
Grok verified both exist via web search (DOI provided).

This asymmetry demonstrates that **listing sources is insufficient** — sources must be actively verified during synthesis.

## Decision

All autonomous learning documents MUST:

## 1. Mandatory Web Verification
Every cited source MUST be verified using the `search_web` or `read_url_content` tool during synthesis. Verification includes:
- Source exists (not hallucinated URL/DOI)
- Source is authoritative for the domain
- Key claims match source content

## 2. Epistemic Status Labels
All claims MUST be tagged:
- **[HISTORICAL]** — Ancient/primary sources
- **[EMPIRICAL]** — Peer-reviewed with DOI/URL (VERIFIED via web tool)
- **[INFERENCE]** — Logical deduction from data
- **[SPECULATIVE]** — Creative synthesis

## 3. Verification Block
Each learning document MUST include:
```markdown
## Source Verification Log
| Source | Verified | Method | Notes |
|--------|----------|--------|-------|
| Hofstadter (2007) | ✅ | Wikipedia/Publisher | Canonical |
| Nature Apr 2025 | ✅ | search_web | DOI:10.1038/... |
```

## 4. Failure Mode
Unverifiable sources MUST be:
- Downgraded to [SPECULATIVE], OR
- Removed from synthesis, OR
- Flagged explicitly: "⚠️ UNVERIFIED: Unable to confirm via web search"

## Consequences

**Positive:**
- Prevents epistemic confidence leaks in autonomous learning
- Makes knowledge quality auditable
- Aligns with Anti-Asch Engine goals (resist conformity bias)
- Eliminates hallucinated sources at the source
- Creates verifiable audit trail

**Negative:**
- Increases time cost per learning session
- Requires network access during synthesis
- Some sources may be paywalled/inaccessible
