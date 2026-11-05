# MISSION-035 pre-mission research notes (2026-11-04)

Status: discovery only. These notes informed the mission draft; none of the mission's phases (P1-P6) have run. Everything below was observed on this machine on 2026-11-04 unless marked UNVERIFIED. Re-verify in P1 before relying on any row.

## Per-backend usage surfaces

| Backend | Surface | Headless? | Cost to read | Window state | Per-task signal | Freshness / caveats |
|---|---|---|---|---|---|---|
| claude | `claude -p "/usage"` (add `--output-format json` for the envelope) | yes | 0 tokens, total_cost_usd 0, num_turns 0 | Current session (5-hour) % used + reset, current week % used + reset | per-session transcript `~/.claude/projects/<proj>/*.jsonl` carries per-message `usage` (input, cache creation/read, output, thinking); interactive /usage adds per-model session tokens and cost | window % is account-wide; breakdown is "approximate, sessions on this machine only"; headless output is human text, parse defensively |
| agy | `agy -p "/usage"` (add `--output-format json`) | yes | 0 tokens (usage.total_tokens 0) | per model group (Gemini models; Claude and GPT models; Gemini 4 Argon): weekly and 5-hour remaining % with reset timestamps | not observed; response `usage` is 0 for the command itself, so per-task tokens must come from the task run's own json output (UNVERIFIED) | quota is shared within a group and consumed proportionally to token cost; Gemini weekly was 12% remaining on 2026-11-04 |
| codex | rollout files `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | yes (file read) | 0 | `rate_limits.primary` (window_minutes 300) and `secondary` (10080) with used_percent, resets_at | `total_token_usage` (input, cached, output, reasoning, total) cumulative per session | lag between activity and the written value UNVERIFIED; `/status` under `codex exec` is NOT a command (sent to the model, cost 18,369 tokens) |
| codex (interactive) | `/usage` then View analytics (tabs 1-6) | no (TUI only, drivable over a pty) | 0 | none on the analytics tabs | tab 6 Chats: per-chat Weekly % and 5-hour % (estimates, may exceed 100%) | Usage tab is "relative units, approximate, may be delayed up to 6 hours"; Messages/Skills tabs showed "Updated Nov 3 11:35 PM UTC"; use as trend context, not deltas |
| copilot | interactive Current tab; `--usage-output-file <file>` | partly (flag exists; unprobed) | UNVERIFIED | plan AI credits: 2,388 / 6,000 AIC (39%) | per-session AIC counter | GATED: cli-agent marks copilot work-only; the screen also showed the repo queued for remote indexing, so check which account this is |
| local MLX (:8080) | response `usage` field; `footprint -p PID` for Metal memory | yes | 0 | none (no quota) | tokens per call | memory, not quota, is the local constraint |

## Hazards observed while researching (each is a constraint in the mission)

1. A `codex exec "/status"` probe was sent to the model and spent 18,369 tokens. Never probe with a prompt.
2. Driving the codex TUI by blind keystrokes is unsafe: a digit sequence reached the "Use this reset?" confirmation (backed out unconfirmed, config unchanged). Codex holds 3 free usage-limit resets (Full: Weekly + 5hr; expiring 6 Nov, 30 Nov, 1 Dec 2026) that are a consumable resource.
3. This codex install currently fails to spawn `codex-code-mode-host` for `codex exec` tool calls (separate defect, not fixed here).
4. Window percentages are account-wide: a mission's own delta is contaminated by any concurrent session, including the orchestrator itself.
5. Auth material lives beside usage data (e.g. `~/.codex/auth.json`); the probe must read named usage fields only.

## What the outputs should become (placement decision)

- `evaluating-agent-work/references/usage-telemetry-sources.md`: the per-backend table above, kept current by the probe (single source of truth), plus `scripts/usage_probe.py` and the measured ledger fields.
- `operating-sanctuary-guardian/references/backend-headroom-and-usage-snapshots.md` (new, short): when in the lease lifecycle a snapshot is taken (checkout, checkin), how a boot digest line "backend headroom" could surface windows, and a pointer to the evaluating-agent-work reference. No duplicated tables.
- The future orchestrator skill (MISSION-033, `orchestrating-sanctuary-missions`): a `references/dispatch-budget.md` that consumes `recommend` and the headroom check before dispatch, pointing here rather than building a second tracker.
