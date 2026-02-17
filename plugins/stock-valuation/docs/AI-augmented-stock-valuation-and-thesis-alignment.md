# Architecture Strategy: AI-Augmented Stock Valuation & Thesis Alignment

**Status:** Draft / Planning
**Date:** 2026-02-13
**Objective:** Leverage the "Investment Logic" (Thesis + Framework) + "Hard Data" (Questrade/YFinance) to create an autonomous Portfolio Analyst.

---

## 1. The Core Concept: "Data-Grounded Reasoning"

The system will not just "chat"; it will perform **Retrieval Augmented Generation (RAG)** specifically tuned for financial logic.

*   **The Brain (Logic):**
    *   *Thesis:* "Twin Revolutions" (`docs/InvestmentThesis/...`)
    *   *Framework:* "Professional Investment Framework v3.1" (`docs/InvestmentFramework/...`)
*   **The Facts (Data):**
    *   *Portfolio:* Real-time holdings via Questrade Sync.
    *   *Market:* Live financials via `yfinance` (PE, Growth, Margins).
*   **The Output:**
    *   Institutional-grade analysis that rigidly adheres to the *Framework's* scoring and the *Thesis's* pillars.

---

## 2. Proposed Architecture

### A. The "Context Builder" Engine (Backend)
We need a robust middleware that prepares the "prompt context" so the LLM doesn't hallucinate.

1.  **Thesis Ingestor**: Parses the Thesis Markdown to extract:
    *   *Pillars* (e.g., "ASI / Compute", "Sovereign Finance")
    *   *Targets* (e.g., "Intel = Sovereign Foundry")
    *   *Risks* (e.g., "18A Node Execution")
2.  **Portfolio Aggregator**:
    *   Flattens holdings from Questrade.
    *   Tags each holding with its "Pillar" (requires a mapping file or LLM classification).
3.  **Financial Enricher**:
    *   Fetches the "Framework Metrics" for each holding:
        *   *Rule of 40* (Rev Growth + EBITDA Margin)
        *   *ROIIC*
        *   *Valuation Multiples* (EV/Sales vs Peers)

### B. The AI "Persona" Layer
We don't just ask "What do you think?". We invoke specific Agents based on the user's mapped prompts:

1.  **Determinist (The Screener)**:
    *   *Role*: Hard math. Calculates Scores (`StartWithBasic`).
    *   *Output*: "NVDA Score: 92/100 (Passes All Rules)"
2.  **Strategist (The Thesis Aligner)**:
    *   *Role*: Qualitative check. "Does INTC still fit the 'Sovereign Foundry' thesis given recent news?"
    *   *Input*: Thesis Doc + News Feed + Price Action.
    *   *Output*: "Thesis Drift Warning: INTC delay rumors threaten 'Sovereign Foundry' pillar."

### C. Hybrid UI Strategy (The "User Experience")

The user asked: *"UI or Chat?"* -> **The answer is BOTH.**

1.  **The Dashboard (The "Heads-Up Display"):**
    *   *Passive Analysis*: A new "Thesis Health" widget.
    *   *Traffic Lights*: Green/Yellow/Red indicators for each holding based on *Framework Alignment*.
    *   *Drift Alert*: "Portfolio Cash (10%) is below Thesis Target (12.7%)."

2.  **The Analyst Studio (The "Deep Dive"):**
    *   *Interactive*: A dedicated view where you can "Summon" the Analyst.
    *   *Actions*:
        *   "Generate Quarterly Report" (Markdown download).
        *   "Stress Test: What if rates rise?" (Scenario analysis).
        *   "Rebalance Recommendation" (Math-based logic).

---

## 3. Implementation Roadmap

### Phase 1: The "Context Pipeline" (Backend)
*   Create `ThesisService.ts`: Reads/Parses the Markdown thesis.
*   Create `AnalysisContext.ts`: Joins specific Stock Data with the relevant Framework Rules.
*   *Deliverable*: A JSON object that *is* the perfect prompt context.

### Phase 2: The LLM Bridge (Agentic Layer)
*   Integrate a local (Ollama) or Cloud (Gemini/Claude) provider.
*   Implement `PromptTemplates`:
    *   `VALUATION_EXPERT_PROMPT`
    *   `THESIS_ALIGNMENT_PROMPT`

### Phase 3: The UI Components
*   **Thesis Widget**: Visual comparison of *Target Allocation* vs *Actual Allocation*.
*   **Stock "Report Card"**: A pop-over on any stock ticker showing its "Framework Score".

---

## 4. Immediate Next Steps (RFC)

**Approve Strategy**: Does this "Hybrid" approach match your mental model?
**Select Pilot**: We are starting with **Stock Valuation** as the pilot implementation.
    *   *Docs*: `docs/architecture/stock-valuation/`
**Next Phase**: **Thesis Alignment & Portfolio Evaluation** will follow.
    *   *Docs*: `docs/architecture/thesis-alignment-and-portfolio-valuation/`
