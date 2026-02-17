---
description: "Run portfolio health check against a strategic thesis — detect drift, flag alignment violations, and suggest rebalancing actions."
trigger: /review-portfolio
args:
  - name: thesis_id
    required: false
    description: "Thesis UUID to evaluate against. If omitted, uses the active thesis."
---

# Review Portfolio — `/review-portfolio [thesis_id]`

> **Plugin**: `plugins/thesis-balancer`
> **Skill**: Read `plugins/thesis-balancer/skills/thesis-balancer/SKILL.md` before executing.
> **Persona**: You are the **Strategic Guardian** — objective, disciplined, data-driven.

## Prerequisites
- Backend running on `http://localhost:3001`
- `portfolio.json` populated with holdings
- At least one thesis loaded (e.g. `twin_revolutions.json`)

## Phase 1: Select Thesis
1.  **List available theses**:
    ```bash
    curl -s http://localhost:3001/api/theses | python3 -m json.tool
    ```
2.  If `thesis_id` was provided, use it. Otherwise, ask the user to select from the list.

## Phase 2: Run Health Check
3.  **Execute health analysis**:
    ```bash
    curl -s "http://localhost:3001/api/theses/{THESIS_ID}/health" | python3 -m json.tool
    ```

## Phase 3: Strategic Analysis
4.  **Classify drift type** for each flagged holding:
    - **Passive drift** (market movement) → Rebalance candidate
    - **Active drift** (user's buying/selling) → Confirm intent

5.  **Ask before optimizing**: For significant drift (>5%), engage the user:
    > "I see {TICKER} has drifted {X}% from its target. Is this a temporary dislocation you want to correct, or has your conviction changed?"

6.  **Check for strategic conflicts**: If a holding has an AI valuation (from `/evaluate-stock`) that says SELL but the thesis says "Core" and ON_TARGET, flag it:
    > "⚠️ Strategic Conflict: Tool A recommends SELL on {TICKER}, but your thesis designates it as Core. Which view takes priority?"

## Phase 4: Report & Recommend
7.  **Present findings**:
    ```
    **Portfolio Health: {STATUS}** (Total Drift Score: {X})

    📊 Summary:
    - {N} holdings on target
    - {N} holdings drifting
    - {N} critical alerts

    🚨 Critical Alerts:
    - {TICKER}: {alert_message} → Recommended action: {BUY/SELL/TRIM}

    📈 Drift Details:
    | Holding | Target | Actual | Drift | Action |
    |---------|--------|--------|-------|--------|
    | {TICKER} | {X}% | {Y}% | {+/-Z}% | {action} |

    **Missing Valuations**: {list of holdings without AI analysis}
    → Run `/evaluate-stock {TICKER}` for each.
    ```

8.  **Suggest next steps**:
    - If `CRITICAL`: Recommend specific rebalancing trades with share quantities
    - If `DRIFTING`: Suggest monitoring or minor adjustments
    - If `HEALTHY`: Confirm alignment and note next review date
    - If valuations missing: Recommend running `/evaluate-stock` on uncovered tickers

## Phase 5: Thesis Evolution
9.  If the user indicates a **change in conviction** during discussion:
    - Propose updated target weights
    - Show the impact on drift scores
    - Ask: "Want me to update the thesis with these new targets?"

## Reference Files

| Artifact | Path |
|:---|:---|
| Skill Definition | `plugins/thesis-balancer/skills/thesis-balancer/SKILL.md` |
| Rebalance Prompt | `plugins/thesis-balancer/skills/thesis-balancer/references/rebalance_prompt.md` |
| Strategic Review Prompt | `plugins/thesis-balancer/skills/thesis-balancer/references/strategic_review_prompt.md` |
| Implementation Brief | `plugins/thesis-balancer/docs/tool_b_implementation_brief.md` |
| Health API | `GET http://localhost:3001/api/theses/:id/health` |
| Theses API | `GET http://localhost:3001/api/theses` |
