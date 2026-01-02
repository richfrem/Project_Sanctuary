# Epistemic Standards for Autonomous Learning
**(Based on ADR 077 & ADR 078)**

## 1. The Core Rule
Autonomous Synthesis must be rigorous. We do not "hallucinate" confidence. If we don't know, we say so.

## 2. Epistemic Status Labels
Every major claim or section in a Learning Document generally requires an Epistemic Status label:

| Label | Definition | Verification Requirement |
|-------|------------|--------------------------|
| **[HISTORICAL]** | Derived from primary/ancient sources. | Cite Source (Author, Date). |
| **[EMPIRICAL]** | Derived from modern peer-reviewed research. | **MANDATORY**: Verify URL/DOI via `search_web`. |
| **[INFERENCE]** | Logical deduction from data points. | Show the reasoning chain. |
| **[SPECULATIVE]** | Creative synthesis/hypothesis. | Explicitly label as "Maybe". |

## 3. Mandatory Source Verification (ADR 078)
You cannot just "list" a source. You must **Verify** it.

### The Verification Loop
1.  **Search**: Use `search_web` to find the paper/article.
2.  **Match**: Confirm Title, Author, and Date match 100%.
3.  **Log**:
    ```markdown
    ## Source Verification Log
    | Source | Verified | Method | Notes |
    |--------|----------|--------|-------|
    | Nature 2025 | ✅ | search_web | DOI:10.1038/... |
    ```

## 4. Truth Anchors & Decay
If a fact is subject to rapid change (e.g., "SOTA Model Performance"), tag it with a **Decay** warning.
- `[DECAY: RAPID]` - Re-verify every 30 days.
