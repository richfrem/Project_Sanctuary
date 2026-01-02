# Mandatory Source Verification for Autonomous Learning

**Status:** APPROVED
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
- **Metadata Match (100%):** Title, Authors, and Date MUST match the source content exactly.
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

## 5. Mandatory Template Schema
All source lists MUST adhere to `LEARNING/templates/sources_template.md`.
- Do not deviate from the schema
- Broken links are strictly prohibited (0% tolerance)

## 6. Mandatory Epistemic Independence (The Asch Defense)
To prevent "Agreement without Independence," all multi-model synthesis MUST declare:
```yaml
epistemic_independence:
  training_overlap_risk: HIGH | MEDIUM | LOW
  data_origin_diversity: [Qualitative Assessment]
  reasoning_path_divergence: [Percentage or Assessment]
```
**Rule:** High agreement with LOW independence MUST be flagged as `[SUSPECT CONSENSUS]`.

## 7. Truth Anchor Temporal Stability
All Truth Anchors MUST include decay metadata to prevent "Zombie Knowledge":
```yaml
truth_anchor:
  anchor_type: empirical | mathematical | procedural | consensus
  decay_mode: none | slow | rapid | unknown
  revalidation_interval: [Days]
```
**Rule:** If `decay_mode` is `unknown` or `rapid`, it MUST NOT be baked into long-term weights (LoRA/Phoenix Forge).

**Rule:** If `decay_mode` is `unknown` or `rapid`, it MUST NOT be baked into long-term weights (LoRA/Phoenix Forge).

## 8. Dynamic Cognitive Coupling (The Edison Breaker)
To resolve the Efficiency vs Integrity tension (LatentMAS vs ASC), systems MUST implement "Dynamic Coupling":
- **Flow State (LatentMAS):** Permitted when SE is within Optimal Range (0.3 - 0.7).
- **Audit State (ASC):** Mandatory when SE indicates Rigidity (<0.2) or Hallucination (>0.8).
**Rule:** The "Edison Breaker" in `operations.py` is the authority for state switching.

## 9. Consequences

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
