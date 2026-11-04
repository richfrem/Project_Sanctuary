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
| claude | `claude -p "/usage"` (0 cost): session (5-hour) and weekly % used + reset | `~/.claude/projects/<repo>*/*.jsonl` per-message `usage`, summed between checkout and checkin | percent is an integer and account-wide |
| agy | `agy -p "/usage" --output-format json` (0 tokens): weekly and 5-hour remaining per model group | pass `--tokens` at checkin from the run's own JSON `usage` | Gemini group shares one weekly limit; Claude/GPT group is separate |
| codex | newest rollout with a `token_count` event in `~/.codex/sessions`: `rate_limits` primary 300 min, secondary 10080 min | `total_token_usage` in the same event | `codex exec "/status"` is NOT a command (it spent 18k tokens); the interactive `/usage` dashboard is relative units and up to 6 h stale: trend only |
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
- **Catalog gap found 2026-11-04** (catalog verified 2026-09-09): it has `claude-sonnet-5` but this session runs `claude-sonnet-5-5`, and `gpt-5.6-luna` but codex config says `gpt-6-luna`; their costs stay null. Using the `claude-sonnet-5` prices as a stand-in priced the real session at $44.79 against the $29.55 Claude reported (+52%), which is why no proxy pricing is allowed. Fix belongs upstream in the cli-agent catalog (Protocol 133), not here.
- Never read auth files (e.g. `~/.codex/auth.json`); every record passes `assert_no_secrets`.
