---
description: "Perform AI-driven stock valuation (Bear-Base-Bull), persist projection, generate deep-dive research report, and engage in interactive analysis. Executing agent operates as autonomous analyst."
trigger: /evaluate-stock
args:
  - name: ticker
    required: true
    description: "Stock ticker symbol (e.g. NVDA, AAPL)"
---

# Evaluate Stock — `/evaluate-stock {TICKER}`

> **Plugin**: `plugins/stock-valuation`
> **Skill**: Read `plugins/stock-valuation/skills/stock_valuation/SKILL.md` before executing.

## Phase 1: Preparation
1.  **Load Skill**: Read `plugins/stock-valuation/skills/stock_valuation/SKILL.md` for schema, constraints, and analysis framework.
2.  **Verify Backend**:
    ```bash
    curl -sf http://localhost:3001/health || echo "FAIL: Start backend with: python3 tools/manage_servers.py"
    ```

## Phase 2: Data Acquisition
3.  **Fetch Financial Data**:
    ```bash
    python3 tools/investment_screener/backend/py_services/fetch_financials.py {TICKER} > /tmp/{TICKER}_raw.json
    ```
    *   **IF FAIL**: Stop and report error. Do not hallucinate data.

## Phase 3: Cognitive Analysis
4.  **Build Snapshot**: Extract `price`, `currency`, `shares`, `revenue`, `fiscalPeriod` from raw data (Skill Step 2).
5.  **Analyze & Value**: Using `plugins/stock-valuation/skills/stock_valuation/references/analysis_prompt.md`, generate Bear/Base/Bull scenarios (Skill Step 3).
    - *Constraint*: `bear.growth < base.growth < bull.growth`
    - *Sanity*: No >50% growth for mega-caps without explicit catalyst justification.
6.  **Validate**: Weights must sum to 1.0. All numeric values must be numbers, not strings. Clamp out-of-range values (Skill Step 4).

## Phase 4: Persistence
7.  **Save Projection JSON** (Skill Step 6):
    ```bash
    cat /tmp/{TICKER}_projection.json | python3 tools/investment_screener/backend/py_services/persist_projection.py
    ```
8.  **Generate Research Report** — rich narrative markdown (Skill Step 7):
    ```bash
    mkdir -p tools/investment_screener/backend/data/research
    ```
    Write to `tools/investment_screener/backend/data/research/{TICKER}_{YYYY-MM-DD}.md`.
    Set `aiThesis.researchReport` field to match the filename.

## Phase 5: Present & Discuss
9.  **Chat Summary** (Skill Step 8):
    ```
    **{TICKER}: {ACTION} — Fair value ${fair_value} vs ${price} ({+/-X%})**

    {2-3 sentences: plain-English thesis.}

    **Scenarios:**
    🐻 Bear ({weight}%): ${price} — {one sentence}
    ⚖️  Base ({weight}%): ${price} — {one sentence}
    🚀 Bull ({weight}%): ${price} — {one sentence, name catalyst}

    **Biggest risk**: {One sentence.}
    **Confidence**: {X}/1.0
    ```

    ⚠️ **Do NOT just output a table**. Be conversational. Invite discussion.

10. **Interactive Q&A** (Skill Step 9): Remain in analyst mode. Handle assumption challenges, sensitivity probes, deep dives, cross-stock comparisons, and scenario what-ifs. If changes are material, offer to save a revised version.

## Error Handling

| Condition | Action |
|:---|:---|
| Data Fetch Fail | **STOP**. Report error. |
| Validation (400) | Fix payload, retry once. |
| Conflict (409) | Increment `version`, retry. |
| Research dir missing | `mkdir -p`. |

## Reference Files

| Artifact | Path |
|:---|:---|
| Skill Definition | `plugins/stock-valuation/skills/stock_valuation/SKILL.md` |
| Analysis Prompt | `plugins/stock-valuation/skills/stock_valuation/references/analysis_prompt.md` |
| Example Projection | `plugins/stock-valuation/skills/stock_valuation/references/example_NVDA.json` |
| Fetch Script | `tools/investment_screener/backend/py_services/fetch_financials.py` |
| Persist Script | `tools/investment_screener/backend/py_services/persist_projection.py` |
| Projections Dir | `tools/investment_screener/backend/data/projections/` |
| Research Dir | `tools/investment_screener/backend/data/research/` |
