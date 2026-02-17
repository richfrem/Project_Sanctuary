# Implementation Brief: Tool B — Thesis Balancer
## Skill + Workflow + Backend + Integration with Tool A

**Author:** Claude (Opus 4.6) — Architecture & Specification  
**For:** Gemini 3 Flash / Gemini 3 Pro (Implementation)  
**Date:** 2026-02-14  
**Status:** Ready for implementation

---

## 1. What You're Building

Tool A (Stock Valuation) answers: **"What is this stock worth?"**  
Tool B (Thesis Balancer) answers: **"Is my portfolio still aligned with what I believe?"**

Tool B takes the user's investment thesis (e.g., "Twin Revolutions"), their actual portfolio holdings, and live market data — then produces a health check showing pillar-level drift, holding-level alerts, and actionable rebalancing trades. It can also invoke Tool A's stock valuation skill for any holding that needs a fresh deep dive.

The user's philosophy: **"You Drive, AI Navigates."**
- User sets the conviction (thesis, pillars, targets)
- AI does the math (drift, scoring, optimization)
- System remembers everything (persisted thesis + projections)

---

## 2. File Map — What to Create and Modify

### New Files

```
.agent/skills/thesis_balancer/
├── SKILL.md                                    # Agent skill definition
└── references/
    ├── thesis_schema.md                        # Schema docs for the thesis JSON
    ├── example_twin_revolutions.json           # The user's thesis as structured data
    ├── health_check_prompt.md                  # LLM prompt for portfolio health analysis
    └── rebalance_prompt.md                     # LLM prompt for trade recommendations

.agent/workflows/thesis-balancer/
└── review-portfolio.md                         # Workflow trigger: /review-portfolio

tools/investment-screener/
├── backend/
│   ├── data/theses/                            # Thesis storage directory
│   │   └── twin_revolutions.json               # The active thesis (created by agent or user)
│   ├── src/
│   │   └── services/
│   │       └── ThesisService.ts                # CRUD + drift calculation + health check
│   └── scripts/
│       └── fetch_portfolio_snapshot.py          # Batch-fetch live prices for all holdings
```

### Files to Modify

```
tools/investment-screener/backend/src/utils/zod-schemas.ts   # Add thesis + health schemas
tools/investment-screener/backend/src/index.ts               # Add thesis API endpoints
tools/investment-screener/frontend/src/services/api.ts       # Add thesis API client
```

---

## 3. The Data Model

### 3.1 Thesis Schema (Zod — add to `zod-schemas.ts`)

This is the core data structure. The user's thesis is stored as structured JSON, not raw markdown. The agent or the user (via UI) can create/edit it.

```typescript
// === THESIS SCHEMAS === (Add to zod-schemas.ts)

export const ThesisHoldingSchema = z.object({
    ticker: z.string().regex(/^[A-Z0-9.\-]{1,10}$/),
    name: z.string().max(100),
    pillarId: z.string(),
    targetWeight: z.number().min(0).max(100),
    thesisForInclusion: z.string().max(2000).optional(),
    thesisBreakers: z.array(z.string().max(500)).max(5).optional(),
    role: z.enum(['core', 'hedge', 'speculative', 'reserve']).default('core'),
});

export const ThesisPillarSchema = z.object({
    id: z.string(),
    name: z.string().max(100),
    targetWeight: z.number().min(0).max(100),
    description: z.string().max(2000).optional(),
    thesisBreakers: z.array(z.string().max(500)).max(5).optional(),
});

export const ThesisSchema = z.object({
    id: z.string().uuid(),
    name: z.string().min(1).max(100),
    schemaVersion: z.literal('1.0'),
    version: z.number().int().nonnegative(),
    createdAt: z.string().datetime(),
    updatedAt: z.string().datetime(),
    description: z.string().max(5000).optional(),
    pillars: z.array(ThesisPillarSchema).min(1).max(20)
        .refine((pillars) => {
            const sum = pillars.reduce((s, p) => s + p.targetWeight, 0);
            return Math.abs(sum - 100) < 0.5;
        }, { message: "Pillar target weights must sum to 100%" }),
    holdings: z.array(ThesisHoldingSchema).min(1).max(100)
        .refine((holdings) => {
            const sum = holdings.reduce((s, h) => s + h.targetWeight, 0);
            return Math.abs(sum - 100) < 0.5;
        }, { message: "Holding target weights must sum to 100%" }),
    globalSettings: z.object({
        driftThresholdPct: z.number().min(0.5).max(20).default(3.0),
        criticalDriftPct: z.number().min(1).max(30).default(5.0),
        rebalanceFrequency: z.enum(['weekly', 'monthly', 'quarterly']).default('quarterly'),
        portfolioValueUSD: z.number().nonnegative().optional(),
    }),
});

export type Thesis = z.infer<typeof ThesisSchema>;
export type ThesisHolding = z.infer<typeof ThesisHoldingSchema>;
export type ThesisPillar = z.infer<typeof ThesisPillarSchema>;
```

### 3.2 Health Check Output Schema

This is what `GET /api/thesis/:id/health` returns. It's the output of Tool B's drift analysis.

```typescript
export const DriftEntrySchema = z.object({
    id: z.string(),
    name: z.string(),
    targetPct: z.number(),
    actualPct: z.number(),
    driftPct: z.number(),           // actual - target (negative = underweight)
    status: z.enum(['ON_TARGET', 'DRIFT', 'CRITICAL']),
});

export const HoldingHealthSchema = DriftEntrySchema.extend({
    ticker: z.string(),
    pillarId: z.string(),
    currentPrice: z.number().optional(),
    marketValue: z.number().optional(),
    role: z.enum(['core', 'hedge', 'speculative', 'reserve']),
    // Cross-reference with Tool A
    hasValuation: z.boolean(),       // Does a projection exist in projections/{TICKER}.json?
    latestAction: z.enum(['BUY', 'HOLD', 'SELL']).optional(),  // From Tool A's aiThesis
    latestFairValue: z.number().optional(),
});

export const HealthCheckSchema = z.object({
    thesisId: z.string().uuid(),
    thesisName: z.string(),
    analyzedAt: z.string().datetime(),
    portfolioValueUSD: z.number(),
    pillarHealth: z.array(DriftEntrySchema),
    holdingHealth: z.array(HoldingHealthSchema),
    alerts: z.array(z.object({
        severity: z.enum(['INFO', 'WARNING', 'CRITICAL']),
        message: z.string(),
        pillarId: z.string().optional(),
        ticker: z.string().optional(),
    })),
    summary: z.object({
        totalDriftScore: z.number(),    // Sum of |drift| across all pillars
        worstPillar: z.string(),
        worstHolding: z.string(),
        overallStatus: z.enum(['ALIGNED', 'DRIFTING', 'CRITICAL']),
    }),
});

export type HealthCheck = z.infer<typeof HealthCheckSchema>;
```

### 3.3 Example Thesis File (`data/theses/twin_revolutions.json`)

Create `references/example_twin_revolutions.json` with this structure. This is the user's actual "Twin Revolutions" thesis converted to structured data:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Twin Revolutions v7.3",
  "schemaVersion": "1.0",
  "version": 1,
  "createdAt": "2026-02-14T12:00:00Z",
  "updatedAt": "2026-02-14T12:00:00Z",
  "description": "The Twin Revolutions thesis: winning the two-front war for ASI supremacy and Sovereign Finance dominance. Builder-First, Empire-Second.",
  "pillars": [
    { "id": "compute",    "name": "ASI / Compute – Chips",         "targetWeight": 27.65 },
    { "id": "cash",       "name": "Cash (Strategic Reserve)",      "targetWeight": 12.67 },
    { "id": "sovfin",     "name": "Sovereign Finance / Digital",   "targetWeight": 12.29 },
    { "id": "titans",     "name": "AI Titans / Cloud",             "targetWeight": 12.40 },
    { "id": "datainfra",  "name": "Data Infra / Supply Chain",     "targetWeight": 10.86 },
    { "id": "power",      "name": "Power / Energy",                "targetWeight":  9.97 },
    { "id": "security",   "name": "Security / Data OS",            "targetWeight":  8.27 },
    { "id": "applied",    "name": "Applied AI / Robotics",         "targetWeight":  5.89 }
  ],
  "holdings": [
    { "ticker": "PSU-U.TO", "name": "Purpose US Cash Fund",    "pillarId": "cash",     "targetWeight": 12.67, "role": "reserve" },
    { "ticker": "INTC",     "name": "Intel Corp",              "pillarId": "compute",  "targetWeight": 10.87, "role": "core",
      "thesisForInclusion": "The Sovereign Foundry: designated US National Champion for onshored compute manufacturing. Core contrarian bet on 18A node.",
      "thesisBreakers": ["18A HVM delay beyond Q4 2026", "No top-5 fabless design win by EOY 2026", "CEO departure without committed IFS successor"] },
    { "ticker": "AVGO",     "name": "Broadcom Inc.",           "pillarId": "compute",  "targetWeight": 6.35,  "role": "core" },
    { "ticker": "NVDA",     "name": "NVIDIA Corporation",      "pillarId": "compute",  "targetWeight": 4.32,  "role": "hedge" },
    { "ticker": "AMD",      "name": "AMD Inc.",                "pillarId": "compute",  "targetWeight": 2.67,  "role": "hedge" },
    { "ticker": "SNPS",     "name": "Synopsys Inc.",           "pillarId": "compute",  "targetWeight": 1.72,  "role": "core" },
    { "ticker": "CDNS",     "name": "Cadence Design",          "pillarId": "compute",  "targetWeight": 1.72,  "role": "core" },
    { "ticker": "GOOG",     "name": "Alphabet Inc.",           "pillarId": "titans",   "targetWeight": 5.86,  "role": "hedge" },
    { "ticker": "MSFT",     "name": "Microsoft Corp.",         "pillarId": "titans",   "targetWeight": 3.60,  "role": "core" },
    { "ticker": "META",     "name": "Meta Platforms",          "pillarId": "titans",   "targetWeight": 2.94,  "role": "core" },
    { "ticker": "ETHA",     "name": "Ethereum ETF",            "pillarId": "sovfin",   "targetWeight": 3.69,  "role": "core" },
    { "ticker": "COIN",     "name": "Coinbase Global",         "pillarId": "sovfin",   "targetWeight": 3.41,  "role": "core",
      "thesisBreakers": ["SEC court victory classifying ETH as security", "Base L2 fails top-3 by volume EOY 2026", "Major hack or loss of customer funds"] },
    { "ticker": "IBIT",     "name": "Bitcoin ETF",             "pillarId": "sovfin",   "targetWeight": 3.13,  "role": "core" },
    { "ticker": "CRCL",     "name": "Circle Internet",         "pillarId": "sovfin",   "targetWeight": 2.06,  "role": "core",
      "thesisBreakers": ["Failure to secure GENIUS Act charter", "USDC peg break below $0.98", "Sustained loss of regulated stablecoin market share"] },
    { "ticker": "VST",      "name": "Vistra Corp.",            "pillarId": "power",    "targetWeight": 4.34,  "role": "core" },
    { "ticker": "CEG",      "name": "Constellation Energy",    "pillarId": "power",    "targetWeight": 4.33,  "role": "core" },
    { "ticker": "OKLO",     "name": "Oklo Inc.",               "pillarId": "power",    "targetWeight": 1.32,  "role": "speculative" },
    { "ticker": "PANW",     "name": "Palo Alto Networks",      "pillarId": "security", "targetWeight": 3.77,  "role": "core" },
    { "ticker": "CRWD",     "name": "CrowdStrike Holdings",    "pillarId": "security", "targetWeight": 2.50,  "role": "core" },
    { "ticker": "ZS",       "name": "Zscaler Inc.",            "pillarId": "security", "targetWeight": 2.00,  "role": "core" },
    { "ticker": "CORZ",     "name": "Core Scientific",         "pillarId": "datainfra","targetWeight": 3.25,  "role": "speculative" },
    { "ticker": "EQIX",     "name": "Equinix Inc.",            "pillarId": "datainfra","targetWeight": 2.82,  "role": "core" },
    { "ticker": "CRWV",     "name": "CoreWeave Inc.",          "pillarId": "datainfra","targetWeight": 1.77,  "role": "core" },
    { "ticker": "ANET",     "name": "Arista Networks",         "pillarId": "datainfra","targetWeight": 1.63,  "role": "core" },
    { "ticker": "VRT",      "name": "Vertiv Holdings",         "pillarId": "datainfra","targetWeight": 1.30,  "role": "core" },
    { "ticker": "HUMN",     "name": "Robotics ETF (HUMN)",     "pillarId": "applied",  "targetWeight": 2.82,  "role": "core" },
    { "ticker": "KOID",     "name": "Robotics ETF (KOID)",     "pillarId": "applied",  "targetWeight": 2.44,  "role": "core" },
    { "ticker": "AIFF",     "name": "Firefly Neuroscience",    "pillarId": "applied",  "targetWeight": 0.63,  "role": "speculative" }
  ],
  "globalSettings": {
    "driftThresholdPct": 3.0,
    "criticalDriftPct": 5.0,
    "rebalanceFrequency": "quarterly",
    "portfolioValueUSD": 150000
  }
}
```

---

## 4. Backend Implementation

### 4.1 `ThesisService.ts`

Create `tools/investment-screener/backend/src/services/ThesisService.ts`. This service handles CRUD for theses and computes health checks. It follows the same patterns as `ProjectionService.ts` (atomic writes, file locking, Zod validation).

**Methods to implement:**

```typescript
class ThesisService {
    // === CRUD ===
    async getThesis(id: string): Promise<Thesis | null>
    async listTheses(): Promise<{ id: string; name: string; updatedAt: string }[]>
    async saveThesis(thesis: Thesis): Promise<void>
    async deleteThesis(id: string): Promise<boolean>

    // === ANALYSIS ===
    async computeHealthCheck(thesisId: string): Promise<HealthCheck>
}
```

**`computeHealthCheck` logic — this is the core of Tool B:**

```
1. Load the thesis from data/theses/{id}.json
2. Load the portfolio from frontend/src/data/portfolio.json (Questrade sync)
3. For each holding in the thesis:
   a. Find the matching position in portfolio.json (by ticker)
   b. Calculate actualWeight = (position.marketValue / totalPortfolioValue) * 100
   c. Calculate drift = actualWeight - targetWeight
   d. Classify: |drift| < driftThreshold → ON_TARGET
                |drift| >= driftThreshold → DRIFT
                |drift| >= criticalDriftPct → CRITICAL
4. Aggregate to pillar level:
   a. For each pillar, sum actualWeights of its holdings
   b. Compare to pillar targetWeight
5. Cross-reference with Tool A:
   a. For each holding ticker, check if projections/{TICKER}.json exists
   b. If it has an AI_AGENT projection, extract latestAction and latestFairValue
   c. Set hasValuation = true/false
6. Generate alerts:
   - CRITICAL: Any pillar with |drift| > criticalDriftPct
   - WARNING: Any holding with drift > driftThreshold AND role = "core"
   - INFO: Holdings with hasValuation = false and role = "core" (suggest running Tool A)
7. Compute summary:
   - totalDriftScore = sum of |pillar drifts|
   - overallStatus = CRITICAL if any pillar is CRITICAL, DRIFTING if any DRIFT, else ALIGNED
```

**Important: No LLM is needed for the health check.** This is pure math — drift calculation, threshold comparison, cross-reference. The LLM is only invoked for the optimization/rebalance step (Step 2 in the interaction flow).

### 4.2 API Endpoints (add to `index.ts`)

```typescript
// === THESIS ROUTES ===

// List all theses
app.get('/api/theses', async (_req, res) => { ... });

// Get a specific thesis
app.get('/api/theses/:id', async (req, res) => { ... });

// Save/update a thesis (Zod validated)
app.post('/api/theses', async (req, res) => { ... });

// Delete a thesis
app.delete('/api/theses/:id', async (req, res) => { ... });

// === HEALTH CHECK (Tool B core) ===

// Compute portfolio health against a thesis — PURE MATH, no LLM
app.get('/api/theses/:id/health', async (req, res) => {
    // Calls thesisService.computeHealthCheck(id)
    // Returns HealthCheck JSON
});

// === OPTIMIZATION (LLM-powered) ===

// Generate rebalancing trades — THIS uses the LLM
app.post('/api/theses/:id/optimize', async (req, res) => {
    // 1. Compute health check first
    // 2. Build prompt with drift data + current prices + Tool A valuations
    // 3. Call Gemini to generate trade recommendations
    // 4. Return structured trade plan
});
```

### 4.3 `fetch_portfolio_snapshot.py`

A batch script that fetches live prices for all tickers in a thesis. Used by the health check when portfolio.json prices are stale.

```python
# Input: JSON array of ticker symbols via stdin or --tickers arg
# Output: JSON object { "NVDA": { "price": 136.21, "currency": "USD" }, ... }
# Uses yfinance batch download for efficiency
```

---

## 5. The Agent Skill

### 5.1 SKILL.md (`.agent/skills/thesis_balancer/SKILL.md`)

Write this file following the same executable-contract pattern as Tool A's SKILL.md. Here is the structure:

```markdown
---
name: thesis_balancer
description: Perform portfolio-level thesis alignment analysis. Computes drift
  between target allocations and actual holdings, generates alerts, and
  optionally produces rebalancing trade recommendations. Can invoke the
  stock_valuation skill for holdings that need fresh analysis.
has_tools: true
---

# Thesis Balancer Skill

## Quick Reference
- **Trigger**: /review-portfolio [thesis_name]
- **Output**: Health check report with pillar/holding drift + optional trade plan
- **Thesis Store**: backend/data/theses/{id}.json
- **Persistence**: GET/POST to http://localhost:3001/api/theses/*
- **Depends on**: stock_valuation skill (Tool A) for per-holding deep dives

## Step 1: Load Thesis
Fetch the active thesis from the backend:
\`\`\`bash
curl -s http://localhost:3001/api/theses | python3 -m json.tool
\`\`\`
If no thesis exists yet, create one from the user's thesis document.
See references/example_twin_revolutions.json for the exact structure.

If the user specifies a thesis name, match by name. Otherwise use the
most recently updated thesis.

## Step 2: Run Health Check
Fetch the computed health check from the backend (this is pure math, no LLM):
\`\`\`bash
curl -s http://localhost:3001/api/theses/{THESIS_ID}/health
\`\`\`

This returns:
- Pillar-level drift (target vs actual weight for each pillar)
- Holding-level drift (target vs actual for each stock)
- Alerts (CRITICAL, WARNING, INFO)
- Cross-references with Tool A valuations (which holdings have AI projections)
- Overall status: ALIGNED / DRIFTING / CRITICAL

## Step 3: Report Health Check to User
Present the results clearly:

### Portfolio Health: {THESIS_NAME}
**Status: {ALIGNED/DRIFTING/CRITICAL}** | Total Drift Score: {X}%

#### Pillar Allocation
| Pillar | Target | Actual | Drift | Status |
|--------|--------|--------|-------|--------|
| ASI/Compute | 27.7% | 22.1% | -5.6% | 🔴 CRITICAL |
| Cash | 12.7% | 5.2% | -7.5% | 🔴 CRITICAL |
| ... | ... | ... | ... | ... |

#### Alerts
- 🔴 CRITICAL: Cash is 7.5% below target (5.2% vs 12.7%)
- 🟡 WARNING: INTC is 5.8% below target (5.1% vs 10.9%)
- ℹ️ INFO: CRWV has no AI valuation — consider running /perform-stock-valuation CRWV

## Step 4: Optimization (Optional — Only If User Requests)
If the user asks to optimize or rebalance, call the optimization endpoint:
\`\`\`bash
curl -s -X POST http://localhost:3001/api/theses/{THESIS_ID}/optimize
\`\`\`

This uses the LLM to generate trade recommendations considering:
- Current drift amounts
- Available cash
- Tool A valuations (prefer buying holdings rated BUY)
- Position sizes needed to close the gap
- Tax implications (prefer adding, not selling)

Report the trade plan to the user.

## Step 5: Deep Dive Integration (Cross-Skill)
If the health check reveals holdings with no AI valuation (hasValuation: false)
and the user wants comprehensive analysis, invoke Tool A for each:

\`\`\`
For each holding where hasValuation == false AND role == "core":
  → Run /evaluate-stock {TICKER}
\`\`\`

This ensures every core holding has a fresh AI projection before the
optimization step runs.

## Step 6: Thesis Maintenance (Optional)
If the user wants to update their thesis (add/remove stocks, change targets):

1. Load the current thesis JSON
2. Apply the requested changes
3. Validate: pillar weights sum to 100%, holding weights sum to 100%
4. POST the updated thesis to the backend
5. Re-run the health check to show the new drift
```

### 5.2 Prompts

**`references/health_check_prompt.md`** — NOT needed for the health check (it's pure math). Only needed for the optimization step.

**`references/rebalance_prompt.md`** — The LLM prompt for generating trade recommendations:

```markdown
# Portfolio Rebalancing Prompt

## Role
You are a portfolio construction specialist implementing a systematic
rebalancing strategy. You do NOT make investment decisions — the user's
thesis defines what they believe. You optimize the MATH to align the
portfolio with the thesis targets.

## Input
You receive:
1. The thesis definition (pillars, targets, holdings)
2. The health check (current drift for every pillar and holding)
3. Available cash and total portfolio value
4. Tool A valuations for holdings that have them (BUY/HOLD/SELL + fair value)

## Rules
1. NEVER recommend selling a position the user has a "core" role for unless
   it's massively overweight (>2x target).
2. Prefer ADDING to underweight positions over selling overweight ones (tax efficiency).
3. If Tool A has a BUY rating on an underweight holding, PRIORITIZE that trade.
4. If Tool A has a SELL rating on an overweight holding, FLAG it but let the user decide.
5. Round trade sizes to whole shares using currentPrice.
6. Do not exceed available cash.
7. Show the BEFORE and AFTER allocation for each recommended trade.

## Output Format
Return strictly formatted JSON:
\`\`\`json
{
  "trades": [
    {
      "action": "BUY",
      "ticker": "INTC",
      "shares": 50,
      "estimatedCost": 1250.00,
      "rationale": "Closes 5.8% underweight in Compute pillar. Tool A rates INTC as BUY with fair value $32 (28% upside).",
      "beforeWeight": 5.1,
      "afterWeight": 9.2
    }
  ],
  "summary": "3 trades totaling $4,500. Reduces total drift from 22% to 8%.",
  "remainingCash": 5500.00,
  "newOverallStatus": "DRIFTING"
}
\`\`\`

## Constraints
- Maximum 10 trades per recommendation
- Minimum trade size: $100
- Do not recommend buying holdings not in the thesis
```

---

## 6. The Workflow

### `review-portfolio.md`

```markdown
---
description: Perform a thesis-aligned portfolio health check. Shows pillar drift,
  holding alerts, and optionally generates rebalancing trades. Can invoke
  stock_valuation (Tool A) for holdings needing fresh analysis.
trigger: /review-portfolio
args:
  - name: thesis
    required: false
    default: latest
    description: Thesis name or ID. "latest" uses most recently updated.
  - name: optimize
    required: false
    default: "false"
    description: If "true", also generate rebalancing trade recommendations.
  - name: deep-dive
    required: false
    default: "false"
    description: If "true", run Tool A valuation for all core holdings missing AI analysis.
---

# Review Portfolio

## What This Does
When you run `/review-portfolio`:
1. Loads your investment thesis (pillar targets, holding targets)
2. Loads your actual portfolio (from Questrade sync or portfolio.json)
3. Computes drift at pillar and holding level
4. Cross-references with Tool A projections (which stocks have BUY/HOLD/SELL ratings)
5. Reports a health check with alerts

With `--optimize true`:
6. Uses LLM to generate specific rebalancing trades

With `--deep-dive true`:
7. Runs /evaluate-stock for each core holding without a fresh AI projection

## Prerequisites
\`\`\`bash
# Backend running
curl -sf http://localhost:3001/health || echo "FAIL: Start with python3 tools/manage_servers.py"

# At least one thesis exists
curl -s http://localhost:3001/api/theses | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d)>0" || echo "FAIL: No thesis found. Create one first."

# Portfolio data exists (from Questrade sync or manual entry)
curl -s http://localhost:3001/api/portfolio | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d.get('items',[]))>0" || echo "FAIL: No portfolio data."
\`\`\`

## Execution
1. **Read Skill**: Load .agent/skills/thesis_balancer/SKILL.md
2. **Load Thesis**: Execute skill Step 1
3. **Health Check**: Execute skill Step 2 (GET /api/theses/{id}/health)
4. **Report**: Execute skill Step 3 — present the drift table and alerts
5. **Optimize** (if --optimize true): Execute skill Step 4
6. **Deep Dive** (if --deep-dive true): Execute skill Step 5 — invoke Tool A for each

## Error Handling
| Error | Action |
|-------|--------|
| No thesis found | Offer to create one from the user's thesis markdown document |
| No portfolio data | Tell user to sync Questrade or manually add holdings |
| Tool A fails for a ticker | Log and skip; continue with remaining holdings |
| Optimize 400 | Zod validation error on thesis; report and fix |
| Backend down | Report to user, suggest restarting |
```

---

## 7. Integration Points with Tool A

This is the key architectural connection. Tool B doesn't just run in isolation — it leverages Tool A's per-stock projections.

### 7.1 Health Check Cross-Reference
When computing the health check, `ThesisService.computeHealthCheck()` reads `data/projections/{TICKER}.json` for each thesis holding. If an `AI_AGENT` projection exists, it extracts:
- `aiThesis.action` → `latestAction` (BUY/HOLD/SELL)
- `aiThesis.fairValue` → `latestFairValue`
- `hasValuation: true`

This data flows into the health check response and informs both the agent's report and the optimization prompt.

### 7.2 Deep Dive Trigger
When the health check finds core holdings with `hasValuation: false`, the agent (or the workflow with `--deep-dive true`) can invoke Tool A:

```
For each holding where hasValuation == false AND role != "reserve":
  /evaluate-stock {TICKER}
```

This runs Tool A's full pipeline (fetch data → analyze → persist) for each missing holding, then re-runs the health check with the fresh data.

### 7.3 Optimization Priority
The rebalance prompt explicitly uses Tool A ratings:
- **BUY-rated underweight** → highest priority trade
- **SELL-rated overweight** → flagged for user decision
- **No rating** → neutral priority, suggest deep dive first

---

## 8. Implementation Order

Follow this sequence. Each step builds on the previous.

### Phase 1: Data Layer (No LLM needed)
1. Add thesis Zod schemas to `zod-schemas.ts`
2. Create `ThesisService.ts` with CRUD methods (same atomic write pattern as ProjectionService)
3. Add thesis API endpoints to `index.ts` (GET/POST/DELETE for theses)
4. Create the `data/theses/` directory
5. Create `example_twin_revolutions.json` — the user's thesis as structured data
6. **Test**: POST the example thesis, GET it back, verify Zod validation

### Phase 2: Health Check (Pure math, no LLM)
7. Implement `computeHealthCheck()` in ThesisService
8. Add `GET /api/theses/:id/health` endpoint
9. Implement the Tool A cross-reference (read projections files)
10. Create `fetch_portfolio_snapshot.py` for live price refresh
11. **Test**: Call health check endpoint, verify drift calculations match manual math

### Phase 3: Agent Skill + Workflow (No LLM yet — just wiring)
12. Write `SKILL.md` following the template in §5.1
13. Write `review-portfolio.md` following the template in §6
14. **Test**: Agent can run `/review-portfolio`, fetch health check, display report

### Phase 4: Optimization (LLM-powered)
15. Write `references/rebalance_prompt.md`
16. Implement `POST /api/theses/:id/optimize` — builds prompt from health check + prices + Tool A data, calls Gemini, returns structured trades
17. **Test**: Run with `--optimize true`, verify trade recommendations are sensible

### Phase 5: Deep Dive Integration
18. Wire up the `--deep-dive` flag to invoke Tool A's `/evaluate-stock` for each missing holding
19. **Test**: Run `/review-portfolio --deep-dive true` on a thesis where half the holdings have no projections

---

## 9. Verification Tests

| # | Test | Expected |
|---|------|----------|
| T1 | POST thesis with pillar weights summing to 99% | Rejected: "weights must sum to 100%" |
| T2 | POST thesis with 101 holdings | Rejected: max 100 |
| T3 | GET /api/theses/{id}/health with aligned portfolio | All pillars ON_TARGET, overallStatus ALIGNED |
| T4 | GET health with INTC at 5% vs 10.87% target | INTC shows CRITICAL drift (-5.87%), Compute pillar shows DRIFT |
| T5 | Health check with NVDA having Tool A BUY projection | holdingHealth for NVDA shows hasValuation:true, latestAction:BUY |
| T6 | Health check with CRWV having no projection | holdingHealth shows hasValuation:false, alert suggests running Tool A |
| T7 | /review-portfolio end-to-end | Agent loads thesis, calls health, displays drift table + alerts |
| T8 | /review-portfolio --optimize true | Agent calls optimize, returns trade recommendations |
| T9 | /review-portfolio --deep-dive true with 5 unvalued holdings | Tool A runs 5 times, health check re-runs with fresh data |
| T10 | Optimize with Tool A BUY on underweight INTC | INTC appears as highest-priority trade in recommendation |
| T11 | Concurrent thesis save + health check | File lock prevents corruption |
| T12 | Delete a thesis, run health check on it | 404, clear error message |
