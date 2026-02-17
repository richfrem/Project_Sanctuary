# Stock Valuation Plugin

> AI-driven stock valuation engine producing Bear/Base/Bull scenarios with persistent projections and deep-dive research reports.

## Commands

| Command | Description |
|:---|:---|
| `/stock-valuation_evaluate-stock {TICKER}` | Run full autonomous valuation workflow |

## Skill

The `stock_valuation` skill provides the analysis framework, schema constraints, and reference prompts used by the agent during cognitive analysis.

## External Dependencies (Web App Scripts)

> **This plugin does NOT own these scripts.** They live inside the Investment Screener web app and are shared between the web app frontend/backend and this agent workflow. Do not move or duplicate them.

| Script | Canonical Path | Purpose |
|:---|:---|:---|
| `fetch_financials.py` | `tools/investment_screener/backend/py_services/fetch_financials.py` | Fetches raw financial data from yfinance |
| `persist_projection.py` | `tools/investment_screener/backend/py_services/persist_projection.py` | Saves projection JSON to the data directory |

| Data Directory | Path |
|:---|:---|
| Projections | `tools/investment_screener/backend/data/projections/` |
| Research Reports | `tools/investment_screener/backend/data/research/` |

## Architecture Docs

| Document | Purpose |
|:---|:---|
| `AI-augmented-stock-valuation-and-thesis-alignment.md` | High-level strategy overview |
| `interaction_flow.md` | User interaction flow |
| `valuation-persistence.md` | How projections are saved and versioned |
| `stock_valuation_sequence.mmd` | Sequence diagram (Mermaid) |

## Dependencies

- `yfinance` (Python) — for fetching financial data
- Backend server running on `localhost:3001`

## Related

- [`thesis-balancer`](../thesis-balancer/) — Portfolio health check and drift analysis
