# MISSION-051: usage prediction that works at small N

Pre-registered 2026-11-05 by claude-code. Status: not started; this file fixes the definitions so the test cannot be tuned afterwards.

## Definitions (fixed now)
- **Size bucket** is assigned by the agent at checkout/checkin from the *expected* billable tokens (input + output + cache writes, as in `tokens_est`): **small** < 50,000; **medium** 50,000-150,000; **large** > 150,000. The *measured* bucket is the same cut points applied to the measured `tokens_est`. Bucket disagreement = declared != measured.
- **Predictor**: per-bucket median of `tokens_est` (and per-bucket-and-model when >= 8 exist), compared with the static per-task ceilings in `track_budget.py`, on held-out tasks chosen by a fixed rule (every 4th task by time order), never tuned.
- **Window calibration**: tokens per percent fitted only on non-concurrent measurements where the 5-hour window moved >= 3 points, or on groups of consecutive small tasks pooled until they sum to >= 3 points. Held-out prediction error is |predicted% - actual%| / actual%.
- **Effort ratio** for one bucket run on two models: (wall seconds, cost, retries) of model B over model A; "noise" = the max/min spread of repeated runs of the cheaper model.

## Data starting point
`LEARNING/research/experiments/mission-035/results-n14.md` and `.agent/learning/events.jsonl` (BUDGET_CONSUMED, source = measured). 14 records exist; 40 are needed with >= 8 per bucket.

## Plan
1. Add a `--size` bucket flag at checkout/checkin (small/medium/large) next to `--task-class`; store it on the record.
2. Run the same medium task on two models (e.g. a bounded inspection on agy and on claude-haiku) at least 3 times each for the effort ratio and the run-to-run spread.
3. Collect passively through real missions to N = 40; then run `usage_probe.py evaluate` and a held-out calibration check as defined above; record the verdict.
