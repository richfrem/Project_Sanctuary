# MISSION-050 v2 results

Frozen candidate; no held-out iteration. V1 files and results preserved.
See `design.md`, `freeze.json` and `audit.json` for parameters and verification.

## Simulation — complete

30 tasks per split; 18 policies; 1,620 trials; 1,200 synthetic ticks per trial.
Same workload distribution as v1, fresh v2 seeds. Prior: 1 interruption / 40 ticks.
Best fixed selected on development: `fixed-steps-2`.

| Split (n=30 adaptive) | Verified adaptive / split-best fixed | Gain vs dev-selected / split-best (>=20%) | Lost acknowledged count / value (<=5%) | Synthetic overhead (<=10%) |
|---|---:|---:|---:|---:|
| development | 2407 / 2827 (`fixed-steps-2`) | -14.8567% / -14.8567% | 2198/4605 = 47.7307% / 47.5939% | 1937/36000 = 5.3806% |
| held_out | 2421 / 2771 (`fixed-time-16`) | -11.6423% / -12.6308% | 2083/4504 = 46.2478% / 46.8218% | 1847/36000 = 5.1306% |
| control | 4606 / 4778 (`end-only`) | 21.1468% / -3.5998% | 0/4606 = 0.0000% / 0.0000% | 1408/36000 = 3.9111% |

Control throughput loss: **3.5998%**, n=30 paired tasks, versus best fixed end-only; threshold <=3%: MISS.

Candidate verdict in the simulation: **FALSIFIED**. Development and held-out
miss completion gain and acknowledged-loss gates; controls miss throughput.
Synthetic overhead passes its tick proxy, but **cannot decide inference-token overhead**.
The simulator supports claims about its exact rollback/resource model only.

## Real-agent arm

Completed 393/630 planned trials; measured usage in completed trials: 214004 server tokens.
30 tasks per split, seven policies; 100 available sequential subgoals per task;
1,000 allocated inference tokens per trial. Score durable, independently verified
subgoals; loss denominator includes every acknowledged successful redo.
All checkpoint requests, including failed/interrupted ones, count in measured overhead.
Allocated budgets are equal; unused tokens remain charged for throughput comparisons.

| Split | Adaptive n | Verified adaptive / best fixed | Gain vs dev-selected / split-best (>=20%) | Ack loss (<=5%) | Measured checkpoint overhead (<=10%) |
|---|---:|---:|---:|---:|---:|
| development | 30 | 13 / 59 (`fixed-steps-1`) | -77.9661% / -77.9661% | 179/192 = 93.2292% | 1248/16381 = 7.6186% |
| held_out | 26 | 17 / incomplete | cannot decide: split incomplete | 157/174 = 90.2299% | 1200/13895 = 8.6362% |
| control | 0 | not measured | cannot decide | cannot decide | cannot decide |

Real interruption-free throughput gate: **cannot decide**, control split incomplete.
Real local-driver candidate verdict: **FALSIFIED** on completed n=30 primary gates. A single completed primary-gate miss is sufficient under the manifest.
The full experiment is **unfinished**. Missing split/threshold results remain **cannot decide**; partial descriptive ratios do not replace the frozen experiment.

### Completed matched task blocks (descriptive only)

| Split | Fully matched tasks | Adaptive / strongest fixed verified | Gain | Ack loss | Token overhead |
|---|---:|---:|---:|---:|---:|
| development | 30 | 13 / 59 (`fixed-steps-1`) | -77.9661% | 179/192 = 93.2292% | 1248/16381 = 7.6186% |
| held_out | 26 | 17 / 55 (`fixed-steps-1`) | -69.0909% | 157/174 = 90.2299% | 1200/13895 = 8.6362% |
| control | 0 | unavailable | cannot decide | unavailable | unavailable |

These are incomplete-sample descriptive comparisons. Their comparator is
retrospective within the matched prefix, not the frozen full-development selection.
They cannot substitute for the planned n=30 per split or establish held-out gates.

The real arm tests local Qwen semantic subgoals, actual measured inference cost,
safe-point context rollback, disposable worker process termination and atomic disk
snapshot recovery. Quota denial is local injection. Vendor handoff is simulated
between agent roles using the same model, so **actual cross-vendor claims remain
undecidable**. No live generation/server is terminated; no production crash durability
or wall-clock fixed-time performance claim follows. Admission byte bounds are
conservative guards, not token telemetry. No population confidence claim is made.

## Prediction review and integration

The frozen prediction that cold-start repair would not resolve loss/completion
tradeoffs was borne out in simulation. Control throughput also missed. Server usage
and lock waits are empirical measurements; supervisor session tokens are not reported.
Worker scope prohibits budget/scoring ledgers, manifests, APM install or flight-board
writes. Orchestrator must record verdicts/ROI and perform installation if integrating.
Control-plane status reported INCONSISTENT (missing DB/hooks); bootstrap was GREEN.
No lifecycle repair outside allowed paths was attempted.
