# Stock Valuation Agent (Tool A)

## Overview
This folder contains the architecture, design, and operational guides for the **Stock Valuation Agent**. This agent is designed to autonomously analyze stock tickers, generate Bear/Base/Bull valuation scenarios based on live financial data, and persist the results to the backend.

## Components

### 1. The Skill
*   **Path**: `.agent/skills/stock_valuation/SKILL.md`
*   **Purpose**: The "Brain". It defines the strict step-by-step procedure the agent must follow.
*   **Capabilities**:
    *   Fetches data via `fetch_financials.py`.
    *   Performs cognitive analysis using `analysis_prompt.md`.
    *   Self-corrects output using `zod-schemas.ts` constraints.
    *   Persists data via HTTP POST to `localhost:3001`.

### 2. The Workflow (Trigger)
*   **Path**: `.agent/workflows/stock-and-portfolio-evaluation/evaluate-stock.md`
*   **Purpose**: The "Router". It connects the user's slash command to the Skill.
*   **Trigger**: `/evaluate-stock {TICKER}`

### 3. References
*   `references/analysis_prompt.md`: The system prompt used for the financial analysis.
*   `references/example_NVDA.json`: A comprehensive 1-shot example of the expected output format.

## How to Use (Antigravity Chat)

To invoke the agent, simply type the slash command in the chat:

```
/evaluate-stock NVDA
```

or

```
/evaluate-stock MSFT
```

### What Happens Next?
1.  **Preparation**: The agent verifies the backend is running and the skill files exist.
2.  **Execution**: It fetches real-time data for the ticker.
3.  **Analysis**: It generates a valuation (Bear/Base/Bull) with rationales.
4.  **Persistence**: It saves the result to `backend/data/projections/{TICKER}.json` with `source: "AI_AGENT"`.
5.  **Reporting**: It posts a summary table and a "Fair Value" assessment back to the chat.

## Architecture & Feedback
*   **Proposal**: [valuation-persistence.md](./valuation-persistence.md)
*   **Security/Red Team**: [red_team_review_round_2_1.md](./red_team_review_round_2_1.md)
*   **Sequence Diagram**: [stock_valuation_sequence.mmd](./stock_valuation_sequence.mmd)
