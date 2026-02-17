# Thesis Balancer Plugin

> Portfolio health monitoring, drift analysis, and thesis alignment enforcement.

## Commands

| Command | Description |
|:---|:---|
| `/thesis-balancer_review-portfolio` | Run portfolio health check against a strategic thesis |

## Skill

The `thesis-balancer` skill provides rebalance and strategic review prompts for evaluating portfolio drift against thesis targets.

## External Dependencies (Web App)

> **This plugin does NOT own the backend or data.** The thesis APIs and data files live inside the Investment Screener web app at `tools/investment_screener/`. Do not move or duplicate them.

| Resource | Canonical Path | Purpose |
|:---|:---|:---|
| Backend Server | `tools/investment_screener/backend/` | Express.js API serving thesis & portfolio endpoints |
| Thesis Data | `tools/investment_screener/backend/data/theses/` | JSON thesis definitions |
| Portfolio Data | `tools/investment_screener/backend/data/portfolio.json` | Current portfolio holdings |
| Projections Data | `tools/investment_screener/backend/data/projections/` | AI valuations (shared with stock-valuation plugin) |

| API Endpoint | Method | Purpose |
|:---|:---|:---|
| `/api/theses` | GET | List available theses |
| `/api/theses/:id/health` | GET | Run health check against a thesis |
| `/api/theses/:id/optimize` | POST | Generate rebalancing trades (future) |

## Architecture Docs

| Document | Purpose |
|:---|:---|
| `tool_b_implementation_brief.md` | Full implementation specification |
| `thesis_alignment_sequence.mmd` | Sequence diagram (Mermaid) |

## Dependencies

- Backend server running on `localhost:3001`
- `portfolio.json` and thesis files loaded

## Related

- [`stock-valuation`](../stock-valuation/) — AI-driven stock valuation workflow
