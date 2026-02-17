# Architecture: Valuation Persistence & AI Agent (v2.0)

This document outlines the design for persisting user-defined valuations and AI suggestions to a permanent backend store, moving beyond the current `LocalStorage` implementation. It combines the original persistence strategy (v1.1) with the new Server-Side AI Valuation Agent architecture.

## 1. Problem Statement
The current `LocalStorage` persistence is browser-bound and lost if the browser cache is cleared. Additionally, we lack a system-level baseline or AI-driven opinion to compare against user projections.
1.  **Persistence**: Stored as version-controlled project data on disk.
2.  **Atomic**: Resist corruption even during hard crashes/power loss.
3.  **Traceable**: Capture exact financial snapshots so projections can be reconstructed later.
4.  **Single Source Bias**: We only store the "User's" version of the truth, missing out on "System" or "AI" opinions.

## 2. Solution Overview

### 2.1 Multi-Source Schema (v2.0)
We will extend the `Projection` schema to support multiple "Authorities" or sources for a single ticker. The JSON file for a ticker (e.g., `NVDA.json`) will become a collection of projections, keyed/tagged by source.

```typescript
type ProjectionSource = 'USER' | 'SYSTEM_YAHOO' | 'AI_AGENT';

interface ValuationBundle {
    ticker: string;
    // The default/active user projection
    userProjection?: Projection; 
    
    // Baseline calculated purely from Yahoo Finance data (mechanical)
    systemBaseline?: Projection;
    
    // Advanced analysis from LLMs
    aiThesis?: Projection & {
        modelName: string; // e.g., "gemini-1.5-pro"
        confidenceScore: number;
        reasoningTrace: string; // Markdown justification
    };
    
    history: Projection[]; // Archived versions
}
```

### 2.2 Storage Model: Per-Ticker Sharding
- **Root**: `backend/data/projections/`
- **File**: `{TICKER}.json` (e.g., `NVDA.json`)
- **Writing**: Every write must be **atomic**. Write to `.tmp` file first, then `fs.renameSync()` to the target.

### 2.3 Server-Side Agent (Skill-Based)
A new **Stock Valuation Skill** (`.agent/skills/stock_valuation/`) will encapsulate the logic for autonomous analysis:
1.  **Ingest**: `scripts/fetch_financials.py` fetches live Yahoo Finance data.
2.  **Analyze**: 
    *   **Tier 1 (Mechanical)**: Calculate purely metric-driven valuations.
    *   **Tier 2 (Cognitive)**: Send data to a Frontier Model (Gemini 3 Pro) to generate Bear/Base/Bull cases + specific justifications.
3.  **Persist**: `scripts/save_valuation.py` writes the result to `backend/data/projections/<TICKER>.json`, merging with existing user data.
4.  **Trigger**: The `.agent/workflows/stock-and-portfolio-evaluation/evaluate-stock` command invokes this skill.

## 3. API Layer & Security

### 3.1 Endpoints (`backend/src/index.ts`)
- `GET /api/projections/:ticker`: Fetch the full valuation bundle (User + AI + System).
- `POST /api/projections`: 
  - **Validation**: Strict schema check on growth, margins, and probability weights (Sum = 1.0).
  - **Conflict Detection**: Backend rejects saves with stale versions (409 Conflict).
  - **Concurrency**: Basic file-locking using `proper-lockfile`.
- `DELETE /api/projections/:ticker/:id`: Remove a specific projection.

### 3.2 Synchronization: API-First
The `storage.ts` service will transition to a **Strict API-First** model:
1.  **Save Flow**: POST to API → On Success → Update LocalStorage cache.
2.  **Migration**: Client-side logic will detect V1.0 (flat) `LocalStorage` data and migrate it to the new schema on first load.

## 4. Implementation Details

### 4.1 CLI Tool Interface
```bash
# Evaluate stock using the new skill
evaluate-stock <ticker> [options]

Options:
  --model <name>       Specific model to use (default: gemini-1.5-flash)
  --save               Persist to backend storage (default: true)
  --force-refresh      Ignore cached data
```

### 4.2 Schema Updates (Zod)
Update `zod-schemas.ts` to include:
*   `source`: Enum `['USER', 'SYSTEM', 'AI']`
*   `aiMetadata`: Optional object for `model`, `tokens_used`, `generated_at`.

### 4.3 UI Enhancements
The `ValuationModeler` will need a "Compare Sources" feature:
*   **Tabs/Overlays**: View "My Projection" vs "Gemini's Projection".
*   **Ghost Sliders**: See AI's recommended sliders as ghost handles on the main UI.

## 5. Security & Verification
- **Input Validation**: Tickers sanitized via `isValidTicker()`.
- **Atomic Renames**: Protect against mid-write crashes.
- **Weight Check**: Sum of bear/base/bull weights MUST be 1.00 ± 0.01.
- **Cost Control**: Backend agents must have rate limits to prevent runaway LLM costs.
