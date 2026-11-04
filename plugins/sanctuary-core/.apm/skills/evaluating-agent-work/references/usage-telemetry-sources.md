# Usage Telemetry: Fuel Gauge and Efficiency (MISSION-035)

Two instruments, both required (a car needs a price-per-litre and km-driven record *and* a fuel gauge):

| Instrument | Question it answers | Source | Command |
|---|---|---|---|
| **Fuel gauge** | How much can I still spend before stalling? | The vendors' own 5-hour and weekly windows | `usage_probe.py headroom`; boot briefing "Fuel gauge" line; `checkout` returns `usage_headroom_warnings` |
| **Efficiency** | What does a unit of work cost in tokens, dollars, time and retries on each model? | Measured at every real checkout/checkin | `track_budget.py status`, `usage_probe.py evaluate` / `calibrate` / `recommend` |

Do not rank models by price per credit alone: a very cheap-per-credit model can take several times the time and effort of a stronger one (right for simple tasks or a nearly empty window, wrong for hard ones). The records carry wall-clock seconds, retries and the quality score so the *effective* cost is measured.

## Sources per backend (verified 2026-11-04; re-verify before relying)
| Backend | Window state | Per-task tokens | Notes |
|---|---|---|---|
| claude | `claude -p "/usage"` (0 cost): session (5-hour) and weekly % used + reset | `~/.claude/projects/<repo>*/*.jsonl` per-message `usage` (input, output, cache read, cache write split into 5-minute and 1-hour TTL), summed between checkout and checkin | percent is an integer and account-wide |
| agy | `agy -p "/usage" --output-format json` (0 tokens): weekly and 5-hour remaining per model group | pass `--tokens` at checkin from the run's own JSON `usage` | Gemini group shares one weekly limit; Claude/GPT group is separate |
| codex | newest rollout with a `token_count` event in `~/.codex/sessions`: `rate_limits` primary 300 min, secondary 10080 min | `total_token_usage` in the same event; at checkin the per-task tokens are computed automatically as the cumulative-total delta across rollouts in the checkout-to-checkin interval (`codex_tokens_between`) | `codex exec "/status"` is NOT a command (it spent 18k tokens); the interactive `/usage` dashboard is relative units and up to 6 h stale: trend only |
| local | none (no quota) | response `usage` field | constraint is memory (`footprint`), not credits |
| copilot | interactive plan AI credits; `--usage-output-file` | n/a | work-only account: excluded unless the human approves a usage-only read |

## How records flow
1. `manage_coordination.py checkout` stores a zero-token snapshot on the lease and caches it for the boot briefing (`.agent/learning/usage_latest.json`, gitignored).
2. `manage_coordination.py checkin` takes a second snapshot and records a measured entry (`track_budget.py record --source measured`: tokens, task class, wall seconds, retries, quality, window deltas, `concurrent`, and `cost_usd` when `--model` is an exact id in the cli-agent catalog). Pass `--task-class`, `--tokens` (non-claude agents), `--quality`, `--model`.
3. Failures never block a checkout/checkin; disable with `--no-usage` or `SANCTUARY_NO_USAGE=1`.

## Reading the gauge honestly
- Windows are account-wide: a delta taken while another session was active is flagged `concurrent` and excluded from calibration.
- Vendors give percentages, not tokens. `usage_probe.py calibrate` estimates tokens-per-percent of the claude 5-hour window by summing tokens and percent over many non-concurrent tasks (single tasks are too coarse); until enough data exists it says `insufficient_data` rather than guessing a runway.
- Prices and model capabilities are NOT kept here: they come from the installed **cli-agent skill's catalogs** (`~/.claude/skills/cli-agent/references/{claude,codex,agy,copilot}-models.json`, maintained per its `model-catalog-maintenance.md`; override the location with `SANCTUARY_CLI_AGENT_REFERENCES`). `usage_probe.py` reads them (`load_prices`) and prices a task only for an EXACT catalog model id, including cache-read and cache-write rates; a model absent from the catalog gets `cost_usd: null`, never a price borrowed from a similarly named model. With vendor-total tokens only (no in/out split) a dollar figure is a lower bound.
- **Catalog gap found and closed 2026-11-04**: the catalog (verified 2026-09-09) lacked `claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-opus-5-5`, `gpt-6-luna`, the GPT-6.1 family and `gemini-4-argon`. It was refreshed upstream under Protocol 133 (companion repo PR #747, installed globally via `apm update --global`). Check: a real Claude Code session on `claude-sonnet-5-5` reported $29.55 and `usage_probe.cost_usd` prices it at $29.49 (4.0k input, 592.9k output, 175.5M cache read, 1.5M 1-hour cache write); the older `claude-sonnet-5` rates as a stand-in had overestimated by 52%. Keep exact-id lookup: when a new model appears and is not in the catalog, its cost stays null until the catalog is refreshed (`cli-agent/references/model-catalog-maintenance.md`).
- Never read auth files (e.g. `~/.codex/auth.json`); every record passes `assert_no_secrets`.
