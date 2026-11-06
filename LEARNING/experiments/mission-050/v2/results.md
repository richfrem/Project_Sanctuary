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

Completed 630/630 planned trials; measured usage in completed trials: 339573 server tokens.
30 tasks per split, seven policies; 100 available sequential subgoals per task;
1,000 allocated inference tokens per trial. Score durable, independently verified
subgoals; loss denominator includes every acknowledged successful redo.
All checkpoint requests, including failed/interrupted ones, count in measured overhead.
Allocated budgets are equal; unused tokens remain charged for throughput comparisons.

| Split | Adaptive n | Verified adaptive / best fixed | Gain vs dev-selected / split-best (>=20%) | Ack loss (<=5%) | Measured checkpoint overhead (<=10%) |
|---|---:|---:|---:|---:|---:|
| development | 30 | 13 / 59 (`fixed-steps-1`) | -77.9661% / -77.9661% | 179/192 = 93.2292% | 1248/16381 = 7.6186% |
| held_out | 30 | 17 / 60 (`fixed-steps-1`) | -71.6667% / -71.6667% | 170/187 = 90.9091% | 1296/16223 = 7.9887% |
| control | 30 | 147 / 147 (`end-only`) | 58.0645% / 0.0000% | 0/147 = 0.0000% | 1248/15512 = 8.0454% |

Real interruption-free throughput loss: **0.0000%**, 147 adaptive / 147 strongest tested fixed durable subgoals, n=30 matched tasks, threshold <=3%: **PASS**.
Against the frozen development-selected `fixed-steps-1` control comparator: 147 / 93, throughput loss -58.0645%. The primary control gate uses the strongest tested fixed control.
Real local-driver candidate verdict: **FALSIFIED** on completed n=30 primary gates. A single completed primary-gate miss is sufficient under the manifest.

### Completed matched task blocks (descriptive only)

| Split | Fully matched tasks | Adaptive / strongest fixed verified | Gain | Ack loss | Token overhead |
|---|---:|---:|---:|---:|---:|
| development | 30 | 13 / 59 (`fixed-steps-1`) | -77.9661% | 179/192 = 93.2292% | 1248/16381 = 7.6186% |
| held_out | 30 | 17 / 60 (`fixed-steps-1`) | -71.6667% | 170/187 = 90.9091% | 1296/16223 = 7.9887% |
| control | 30 | 147 / 147 (`end-only`) | 0.0000% | 0/147 = 0.0000% | 1248/15512 = 8.0454% |

These matched-block comparisons use the retrospective strongest fixed within
each displayed cohort. The primary table also reports the frozen development-selected
comparator. An incomplete cohort cannot establish the planned n=30 held-out gates.

### Adaptive policy interpretation

The real adaptive policy made **0 early checkpoint requests** under the frozen design.
With zero early requests it degenerated to end-only: any checkpoint was a final flush.
This is a **valid negative finding for this frozen design**, not an implementation defect
in translating the registered policy. The independent audit checks each decision against
the frozen formula. The design couples a 100-token initial checkpoint-cost prior with
a nine-cost overhead floor, requiring at least 900 work tokens before a first early
request; conservative admission and final flushing stop work earlier in these short
1,000-token trials. This is a design feasibility limitation. It does not test an
adaptive scheduler operating with learned checkpoint costs or meaningful early decisions.
No policy, prior, budget, threshold, prompt or seed was tuned after seeing these results.

### Claims supported by each arm

| Hypothesis claim | Simulation | Real-agent arm |
|---|---|---|
| >=20% durable completion gain and <=5% acknowledged loss | Tests these claims within the frozen synthetic rollback/value model; misses refute this candidate there. | Completed n=30 interrupted splits test these claims for this local Qwen driver; misses refute this candidate there. |
| <=10% checkpoint inference-token overhead | Cannot support it: tick overhead is only a proxy. | Tests pooled server-reported prompt+completion checkpoint overhead, including failed calls; pooled compliance does not establish worst-task compliance. |
| <=3% interruption-free throughput loss | Tests the synthetic control claim; 3.5998% misses. | Completed n=30 real controls test the local-driver claim at equal allocated token budgets; incomplete controls cannot decide it. |
| Adaptive early scheduling improves real agent recovery | Synthetic scheduling decisions can be examined, but cannot establish real model recovery. | Cannot support this mechanism claim: zero early requests reduce adaptive to end-only. |
| General production, cross-vendor, live-generation or real quota-outage recovery | Cannot support these claims. | Cannot support these claims: safe-point driver recovery only; handoff and quota denial were simulated. |

Neither arm establishes universal optimality over every fixed interval or a population
effect beyond its finite tasks. Fresh v2 seeds also prevent a paired causal claim that
the cold-start change itself caused any difference from v1.

The real arm tests local Qwen semantic subgoals, actual measured inference cost,
safe-point context rollback, disposable worker process termination and atomic disk
snapshot recovery. Quota denial is local injection. Vendor handoff is simulated
between agent roles using the same model, so **actual cross-vendor claims remain
undecidable**. No live generation/server is terminated; no production crash durability
or wall-clock fixed-time performance claim follows. Admission byte bounds are
conservative guards, not token telemetry. No population confidence claim is made.

## Prediction review and integration

The frozen prediction that cold-start repair would not resolve loss/completion
tradeoffs was borne out in simulation. Simulation control throughput also missed.
The real control result is reported separately above. Server usage
and lock waits are empirical measurements; supervisor session tokens are not reported.
Worker scope prohibits budget/scoring ledgers, manifests, APM install or flight-board
writes. Orchestrator must record verdicts/ROI and perform installation if integrating.
Control-plane status reported INCONSISTENT (missing DB/hooks); bootstrap was GREEN.
No lifecycle repair outside allowed paths was attempted.


## Reviewer-required disclosures (added after the final independent review; see orchestrator-addendum.md)
- The v2 prior (1 interruption per 40 ticks, 400 tokens in the real arm) was chosen after seeing v1 held-out results: v2 is a post-hoc repair of the cold-start defect, not a clean held-out replication. Fresh task seeds only prevent reuse of v1 held-out instances; the statement above that nothing was tuned after seeing these v2 results remains true.
- About 46.1% of the allocated inference budget is unused (339,573 of 630,000 tokens across 630 completed trials); the runner ends a trial at its first final flush.
- Wrong work calls are mostly deterministic repeats: 1,113 of 1,282 repeat an already-wrong (trial, cursor, state, output).
- Interrupted attempt: 2 calls (113 tokens) of `control-230014-fixed-steps-1` are interrupted-attempt overhead, excluded from that trial's totals and from the checkpoint-overhead figures above; see interrupted-attempts.json.
- Control gates: the adaptive policy was behaviourally identical to `end-only` (identical results in all 90 paired tasks), so the 0.0% real control loss passes by construction; by request service time the loss is 1.09% (pass).
- The real-arm verdict is for this frozen design and candidate. At this budget and schedule even the best fixed policy loses 57-58% of acknowledged progress, so <=5% loss was unreachable for every tested policy.
