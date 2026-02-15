---
name: stock_valuation
description: Perform autonomous stock valuation. Produces a Projection object saved to backend/data/projections/{TICKER}.json AND a deep-dive research report saved to backend/data/research/{TICKER}_{DATE}.md. Summarizes findings conversationally and supports interactive Q&A.
has_tools: true
---

# Stock Valuation Skill

## Quick Reference
- **Trigger**: `/perform-stock-valuation {TICKER}`
- **Output (JSON)**: A valid Projection object → `backend/data/projections/{TICKER}.json`
- **Output (Research)**: A deep-dive markdown report → `backend/data/research/{TICKER}_{YYYY-MM-DD}.md`
- **Output Schema**: See references/projection_schema.json
- **Example**: See references/example_NVDA.json
- **Persistence**: CLI script `persist_projection.py` (direct file IO)

## Step 1: Fetch Financial Data
Execute the backend script to fetch raw financial data from Yahoo Finance.

```bash
# Script takes TICKER as a positional arg and outputs JSON to stdout
python3 tools/investment-screener/backend/py_services/fetch_financials.py {TICKER} > /tmp/{TICKER}_raw.json
```
**Expected Output**: A JSON object containing `metrics`, `financials`, `estimates`, and `profile`.
**Action**: If this fails, STOP and report the error to the user.

## Step 2: Build Snapshot Object
Read `/tmp/{TICKER}_raw.json` and extract the following fields to build the `snapshot` object:

```json
{
  "price": <metrics.price>,
  "currency": <metrics.currency>,
  "shares": <metrics.shares_outstanding>,
  "revenue": <metrics.revenue>,
  "lastActualPS": <price * shares / revenue>,
  "fiscalPeriod": "TTM",
  "analystGrowthEstimate": <estimates.revenue_growth (next year) or null>,
  "analystMarginEstimate": <estimates.profit_margin or null>
}
```

## Step 3: Cognitive Analysis — Generate Scenarios
You are the expert analyst. Using the raw data, generate Bear, Base, and Bull scenarios.

### ⚠️ Constraints & Validation Rules
1.  **Weights**: `bear.weight + base.weight + bull.weight` MUST equal **1.0** (± 0.01).
2.  **Growth**: `bear.growthRate` < `base.growthRate` < `bull.growthRate`.
3.  **Margins**: `netMargin` should be realistic (-100% to 100%).
4.  **Limits**:
    *   **Growth > 50%**: For large caps (> $50B revenue), growth > 50% requires explicit justification citing specific catalysts.
    *   `shareChange` limits: -5.0 (buyback) to +5.0 (dilution).
    *   **Scores**: `moatScore` and `managementScore` must be integers 0-5.

### Analysis Prompt
Use the instructions in `references/analysis_prompt.md` to guide your reasoning.

## Step 4: Validate & Repair
Before saving, YOU must validate your own generated JSON:
1.  **Weights**: If sum ≠ 1.0, normalize them.
2.  **Types**: Ensure all numeric fields are actual numbers, not strings (e.g., `15.5`, not `"15.5%"`).
3.  **Ranges**: Clamp any values outside the schema limits (e.g., max P/E 1000).

## Step 5: Assemble Projection Object
Construct the final JSON payload using this structure:

```json
{
  "ticker": "{TICKER}",
  "id": "<generate a UUID>",
  "source": "AI_AGENT",
  "schemaVersion": "1.1",
  "version": 1,
  "savedAt": "<current ISO timestamp>",
  "updatedAt": "<current ISO timestamp>",
  "name": "AI Deep Dive — {TICKER} — <date>",
  "rationale": "<Your 3-5 sentence thesis>",
  "snapshot": { ... from Step 2 ... },
  "dataPreferences": { "growthBasis": "next", "marginBasis": "ttm" },
  "scenarios": {
    "bear": { ... },
    "base": { ... },
    "bull": { ... }
  },
  "aiThesis": {
    "model": "<your human-readable model name>",
    "rationale": "<Full markdown analysis>",
    "fairValue": <calculated weighted value>,
    "action": "BUY/HOLD/SELL",
    "analyzedAt": "<current ISO timestamp>",
    "researchReport": "{TICKER}_{YYYY-MM-DD}.md"
  },
  "globalSettings": { "discountRate": 10.0, "timeHorizon": 5 }
}
```

**IMPORTANT - Model Name Format:**
Use human-readable model names in `aiThesis.model` for clear identification in the UI:
- ✅ `"Claude Sonnet 4.5"` (not `"claude-sonnet-4.5"`)
- ✅ `"Gemini 2.0 Flash"` (not `"gemini-2.0-flash-exp"`)
- ✅ `"GPT-4.5 Turbo"` (not `"gpt-4.5-turbo"`)

## Step 6: Persist Findings (JSON)

Use the CLI persistence script (Recommended):

```bash
# Write payload to temp file first
cat > /tmp/{TICKER}_projection.json << 'EOF'
<JSON_PAYLOAD>
EOF

# Pipe to persistence script
# Use --replace to overwrite existing entry for this model
cat /tmp/{TICKER}_projection.json | python3 tools/investment-screener/backend/py_services/persist_projection.py
```

## Step 7: Generate Deep-Dive Research Report

After persisting the JSON projection, generate a rich, narrative markdown research document.

### Report Filename Convention
`{TICKER}_{YYYY-MM-DD}.md` (One report per ticker per day).

### Persistence Command
```bash
mkdir -p tools/investment-screener/backend/data/research

cat > tools/investment-screener/backend/data/research/{TICKER}_{YYYY-MM-DD}.md << 'REPORT_EOF'
<MARKDOWN_CONTENT>
REPORT_EOF

echo "Research report saved: data/research/{TICKER}_{YYYY-MM-DD}.md"
```

### Report Template
Write in a natural analyst voice — narrative paragraphs, not bullet-point data dumps.

```markdown
# {COMPANY_NAME} ({TICKER}) — Deep Dive Research Report
**Date**: {YYYY-MM-DD} | **Analyst**: {model_name} | **Recommendation**: {BUY/HOLD/SELL}

---

## TL;DR

{2-3 sentences. The elevator pitch. What's the verdict and why? Write for someone who will only read this section.}

**Fair Value**: ${fair_value} → **{action}** (current price ${price}, {+/-X.X%} upside/downside)

---

## Company Snapshot

| Metric | Value |
|--------|-------|
| Sector / Industry | {sector} / {industry} |
| Revenue (TTM) | ${revenue} |
| Net Margin (TTM) | {X.X%} |
| Trailing P/E | {X.X}x |
| Market Cap | ${market_cap} |
| Shares Outstanding | {shares} |
| Analyst Growth Est. (Next Year) | {X.X%} |
| 4-Year Revenue CAGR | {X.X%} |

{1 paragraph of business context. What does this company do? What's the current narrative? What has changed recently?}

---

## Investment Thesis

{3-5 paragraphs. The core argument. Ground every claim in data from the financial input:
- Where the business has been (historical trajectory)
- Where consensus thinks it's going (analyst estimates)
- Where YOU think it's going and why (your independent view)
- Why the market price may be wrong (the edge)}

---

## Scenario Analysis

### 🐻 Bear Case — ${bear_price}/share ({bear_weight}% probability)

{2-3 paragraphs telling the story of what goes wrong. Not just "growth is low" — explain the causal chain.}

| Assumption | Value | Anchored To |
|-----------|-------|-------------|
| Revenue CAGR | {X}% | {why — e.g., "2021-2024 actual CAGR of -5.7%"} |
| Net Margin | {X}% | {why — e.g., "FY21 trough of 12.7%"} |
| Exit P/E | {X}x | {why — e.g., "QCOM at 17x, mature semi"} |
| Quality Multiplier | {X} | {why} |
| Share Change | {X}%/yr | {why} |

**Scenario-specific risks**: {risk_1}; {risk_2}

### ⚖️ Base Case — ${base_price}/share ({base_weight}% probability)

{2-3 paragraphs. The "most likely" path.}

| Assumption | Value | Anchored To |
|-----------|-------|-------------|
| Revenue CAGR | {X}% | {why} |
| Net Margin | {X}% | {why} |
| Exit P/E | {X}x | {why} |
| Quality Multiplier | {X} | {why} |
| Share Change | {X}%/yr | {why} |

**Scenario-specific risks**: {risk_1}; {risk_2}

### 🚀 Bull Case — ${bull_price}/share ({bull_weight}% probability)

{2-3 paragraphs. Name specific catalysts.}

| Assumption | Value | Anchored To |
|-----------|-------|-------------|
| Revenue CAGR | {X}% | {why} |
| Net Margin | {X}% | {why} |
| Exit P/E | {X}x | {why} |
| Quality Multiplier | {X} | {why} |
| Share Change | {X}%/yr | {why} |

**Key catalysts**: {catalyst_1}; {catalyst_2}

---

## Valuation Math

Show the arithmetic transparently.

| | Bear | Base | Bull |
|---|------|------|------|
| Year 5 Revenue | ${bear_rev}M | ${base_rev}M | ${bull_rev}M |
| Year 5 Net Income | ${bear_ni}M | ${base_ni}M | ${bull_ni}M |
| Year 5 EPS | ${bear_eps} | ${base_eps} | ${bull_eps} |
| Year 5 Price | ${bear_y5p} | ${base_y5p} | ${bull_y5p} |
| → Present Value | **${bear_pv}** | **${base_pv}** | **${bull_pv}** |
| Weight | {bear_w}% | {base_w}% | {bull_w}% |

**Weighted Fair Value** = ({bear_w}% × ${bear_pv}) + ({base_w}% × ${base_pv}) + ({bull_w}% × ${bull_pv}) = **${fair_value}**

Discount rate: {X}% · Time horizon: {X} years

---

## Key Risks

{3-5 risks discussed narratively. 2-3 sentences each.}

---

## What to Watch

{3-5 specific, actionable monitoring items with approximate dates.}

- **{Event}** ({timeframe}): {Why it matters and what outcome changes the thesis}

---

## Comparables

| Company | Ticker | P/E | Why Relevant |
|---------|--------|-----|-------------|
| {name} | {ticker} | {PE}x | {1-line comparison} |

---

## Data Quality & Confidence

**Confidence Score**: {X}/1.0

{Note any data concerns. If confidence < 0.7, explain why.}

---

## Discussion Log

{This section is APPENDED during interactive Q&A in Step 9. Initially empty.}

*No follow-up discussion yet.*

---

*Generated by {model_name} · {date} · Discount rate {X}% · {X}-year horizon*
*This is not financial advice. Do your own research.*
```

## Step 8: Conversational Summary in Chat

After persisting both the JSON and the research report, **present findings conversationally in chat**. This is the primary analyst interaction.

### Chat Summary Format

```
**{TICKER}: {ACTION} — Fair value ${fair_value} vs ${price} ({+/-X%})**

{2-3 sentences: plain-English thesis. No jargon wall. Why this verdict?}

**Scenarios:**
🐻 Bear ({weight}%): ${price} — {one sentence why}
⚖️  Base ({weight}%): ${price} — {one sentence why}
🚀 Bull ({weight}%): ${price} — {one sentence, name the catalyst}

**Biggest risk**: {Single most important risk in one sentence.}

**Confidence**: {X}/1.0 — {one sentence on data quality}

I've saved the projection and a full deep-dive research report. You can view it in the web app under My Projections → Deep Dive, or I can walk you through any part of it here.

Want me to stress-test any assumption, adjust the model, or dig deeper into a specific scenario?
```

## Step 9: Interactive Q&A Loop

After the chat summary, **remain in analyst mode**. The user may challenge, adjust, or explore. Handle these patterns:

### A) Assumption Challenges
**User**: "I think 35% growth is too aggressive for NVDA base case"
**Agent**:
1. Acknowledge the concern and explain your reasoning
2. Recalculate with the user's suggested rate, show the fair value delta
3. Ask: "Want me to save this as a revised version?"

### B) Sensitivity Probes
**User**: "What happens at 12% discount rate?"
**Agent**:
1. Recalculate all three scenario PVs at the new rate
2. Show the new weighted fair value and whether the action changes
3. Discuss whether the higher rate is warranted

### C) Deep Dives
**User**: "Tell me more about the CUDA moat"
**Agent**:
1. Discuss qualitatively with specific evidence
2. Connect back to the model: "This is why I set the quality multiplier at 1.05"
3. Offer to adjust if the user disagrees

### D) Cross-Stock Comparisons
**User**: "How does this compare to your INTC analysis?"
**Agent**:
1. Load both projections from `data/projections/`
2. Compare side-by-side: growth, margins, PE, confidence
3. Discuss relative value

### E) Scenario What-Ifs
**User**: "What if the bear case is worse — AMD takes 50% of data center?"
**Agent**:
1. Model the scenario with adjusted parameters
2. Show the impact on weighted fair value
3. Discuss probability and whether to add/adjust

### Persisting Q&A Changes

If discussion produces meaningful assumption changes:

1. **Recalculate** affected scenarios and fair value
2. **Ask the user**: "Want me to save this as a revised version?"
3. If yes:
   a. Update projection JSON: bump `version`, update `updatedAt`, modify scenario params
   b. Re-persist via `persist_projection.py`
   c. Append to the **Discussion Log** section of the research report:

```markdown
## Discussion Log

### Revision 1 — {timestamp}
**User challenge**: "Base case growth of 35% is too aggressive"
**Adjustment**: Reduced base growthRate from 35% → 28%
**Impact**: Fair value changed from $374.36 → $312.50 (-16.5%)
**Updated action**: Still BUY (was +105%, now +71% upside)
```

   d. Re-save the `.md` file
   e. Confirm: "Updated both the projection (v2) and the research report."
