# MISSION-071: OpenRouter Micro-Tier End-to-End Smoke via cli-agent

**Verdict: CORROBORATED** (2026-11-10, 3 of 3 runs)

## Question
After moving the JEV router and OpenRouter client out of this repo into the companion `cli-agent` skill, does a Sanctuary mission spec that pins a cheap OpenRouter model still run start to end through the installed skill alone?

## Design (frozen before runs; hashes in `freeze.json`)
- Spec: `task.json` (sort three JSON keys; pinned `meta-llama/llama-3.2-3b-instruct`; `expected_output` given).
- Evaluator: installed `~/.claude/skills/cli-agent/scripts/smoke_test_openrouter.py`, local fallback disabled.
- Runner: `run.py`, three consecutive runs, per-run cost ceiling $0.0001.
- Pass requires all four checks per run: route matches spec, live OK with no fallback, output equals expected and keys sorted, spend ledger incremented.

## Results (`results.json`)
| Run | Route | Live | Output | Ledger | Tokens in/out | Cost |
|---|---|---|---|---|---|---|
| 1 | PASS | PASS | PASS | PASS | 100 / 21 | $0.0000092 |
| 2 | PASS | PASS | PASS | PASS | 100 / 21 | $0.0000092 |
| 3 | PASS | PASS | PASS | PASS | 100 / 21 | $0.0000092 |

Total cost: $0.0000276.

## Limits
One trivial task, one model. This proves the installed consumer path works; it says nothing about router accuracy on varied tasks (see the MISSION-070 caveat in cli-agent's `jev-router.md`).

## Reproduce
```bash
python3 LEARNING/experiments/mission-071/run.py
```
