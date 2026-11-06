# MISSION-050 codex-b checkpoint handoff

## 1. Changed files

- New `LEARNING/experiments/mission-050/v2/`: preregistration, frozen fixtures and
  hashes, simulation runner/evidence, real-Qwen runner/raw usage/responses,
  snapshots, checker, test output, derived reports, operational deadline guard,
  accounting, investment review, remaining-work list and final hash seal.
- Guardian `scripts/adaptive_checkpoint.py`: separate v2 policy with prior-aware
  runner; v1 class/default semantics remain unchanged for exact replay.
- Guardian `tests/test_adaptive_checkpoint.py`: v2 cold-start/prior/causality tests.
- Guardian `references/adaptive-interruption-policy.md`: v2 findings and limits.

V1 experiment artifacts are byte-for-byte unchanged (freeze + final audit).
`LEARNING/missions/MISSION-050.json` was already modified on arrival and was not
edited by this worker. No manifest, index, backlog, ledger, git or APM mutation.

## 2. Summary

Recommendation: **FALSIFIED for this frozen v2 candidate**. The complete v2
simulation misses gain, acknowledged-loss and interruption-free throughput.
The complete n=30 real development arm misses gain and acknowledged loss.
A completed primary gate miss is sufficient under the manifest. However, the
FULL real experiment is unfinished; held-out requirements and real controls
must not be represented as completed or corroborated.

The cold-start fix uses a stated pseudo-count prior, not held-out schedule
knowledge. The frozen immutable observation boundary is unchanged. Every source
and fixture hash still matches the pre-run freeze. No held-out tuning occurred.

## 3. Exact commands and outcomes

See `commands.md` for pasteable exact historical/resume commands.
- Simulation: 1,620 trials, 30 tasks/split, 18 policies, 5.477936082985252 seconds.
- Tests: 36 passed in 1.79s (`validation-all.txt`).
- Final audit: all 1,620 simulation trials replayed exactly; all 393 completed
  real trials/3,551 actual response IDs audited; frozen/v1 hashes PASS;
  acknowledgment/resource/atomic recovery and policy-decision checks PASS;
  zero partial measured trials and zero inference-budget overruns.
- Real inference: 214,004 measured prompt+completion tokens; 556 rejected work
  answers (not acknowledged). No self-reported supervisor token count.
- Lock wait: 2355.892331 measured seconds across requests; request wall time:
  4020.562378 seconds. Shared access cleared around 05:29Z, then became contended
  again around 05:54Z. The 90-minute round's time constraint plus this measured
  shared-lock delay prevented full inference execution; the endpoint was active.
- Runner exit 1 was intentional: the next-trial snapshot guard stopped execution
  before the next request. `checkpoint-stop.json` identifies the explicit marker;
  it was removed AFTER runner exit. The client runner, its children, supervisor
  and progress monitor have exited. The human model server was never managed.

## 4. Evidence paths

- `design.md`, `freeze.json`: pre-run design/prior/prompt/source/fixture freeze.
- `simulation-evidence.json.gz`, `simulation-summary.json`: complete simulation.
- `real-fixtures.json`, `real-selected-fixed.json`: frozen tasks/schedules and
  development-selected comparator, written before held-out execution.
- `real-calls.jsonl`: complete raw requests, outputs, server usage and lock times.
- `real-events.jsonl`, `real-trials.jsonl`, `snapshots/`: safe-point interruptions,
  terminated/replaced PIDs, denied requests, receipts and durable snapshots.
- `audit-final.txt`, `audit.json`, `verify_v2.py`: independent semantic checker,
  exact simulation replay, model-usage accounting and policy/commit replay.
- `results.md`, `real-matched-cohorts.json`, `verdict.json`: scoped thresholds.
- `final-accounting.json`, `investment-review.json`, `seal.json`: actual measured
  expenditure, provisional five-dimension rubric and final artifact hashes.

## 5. Thresholds with exact scope

Full numerator/denominator tables are in `results.md`; no simulation ticks are
represented as inference tokens. All real subgoal values are 1, so count/value
loss fractions are equal there.

| Arm/scope | Gain >=20% | Ack count/value loss <=5% | Overhead <=10% |
|---|---:|---:|---:|
| Simulation development, n=30 | 2407/2827 - 1 = -14.8567% MISS | 2198/4605 = 47.7307% / 47.5939% MISS | 1937/36000 = 5.3806% tick proxy PASS; token gate cannot decide |
| Simulation held-out, n=30 | -11.6423% vs dev-selected; -12.6308% vs split-best MISS | 2083/4504 = 46.2478% / 46.8218% MISS | 1847/36000 = 5.1306% tick proxy PASS; token gate cannot decide |
| Real development, n=30 | 13/59 - 1 = -77.9661% MISS | 179/192 = 93.2292% MISS | 1248/16381 = 7.6186% pooled PASS |
| Real held-out matched prefix, n=26/30, descriptive ONLY | 17/55 - 1 = -69.0909% | 157/174 = 90.2299% | 1200/13895 = 8.6362% |

Fourth threshold: simulation controls n=30, 4606 adaptive / 4778 best fixed,
throughput loss **3.5998% >3% MISS**. Real controls n=0: **cannot decide**.
Real worst-task overhead is 11.3475%, and worst-task acknowledged loss is 100%
in both measured adaptive cohorts; pooled overhead passes do not imply every-task
compliance. Held-out data do not meet the required n=30; full held-out gates remain
undecided. An extra fixed-time-480 held-out trial exists beyond the matched prefix
(seed 230026); it is preserved but excluded from n=26 matched comparisons.

## 6. Defects and uncertainties

- V2's nonzero hazard fixes simulation cold start but does not meet gain/loss or
  control requirements. Fresh v2 seeds prevent a paired causal v1/v2 estimate.
- Real adaptive made zero nonfinal checkpoint requests: the 9-cost floor and
  initial 100-token cost require >=900 work tokens before the first request,
  while conservative admission/final flushing stops earlier. It therefore
  reduces to end-only in this short-budget arm and does not learn a measured
  checkpoint cost for early scheduling. Do NOT fix/tune this held-out run.
- Semantic model failures are genuine measured misses. Example: sorting '49'
  returned '04'. Failed calls/retries consume usage and earn no acknowledgment.
- Four event kinds are frozen for every interrupted schedule, but only crossed
  token boundaries fire. Exact scheduled/observed kind counts are in accounting.
- Worker termination kills a real disposable client after a completed call, not
  the server or live generation. Context recovery is explicit driver state.
  Quota denial is local injection. Handoff uses the same Qwen model in a fresh
  client/role. Actual vendor recovery, long quota outages, arbitrary context
  reset, mid-generation recovery and production crash durability remain untested.
- Fixed-time is consumed-token time, not wall-clock time. The fixed grid is a
  finite tested set, not a universal optimum. Byte admission bounds are guards,
  not reported usage; substantial unused allocation is reported explicitly.
- Control-plane status was INCONSISTENT (DB/hooks absent); repair is outside
  scope. APM install and ledger updates are also reserved for the orchestrator.

## 7. Remaining work

Exactly **237 inference trials**: **27 held-out + 210 real controls**.
See `remaining.json` for every split/seed/policy and execution order. Next trial:
`held_out-230026-adaptive-v2`. Complete the remaining six policies of held-out
seed 230026 and all seven for seeds 230027..230029; then all 30 control tasks
across all seven policies. Simulation/offline verification is complete.

Do not truncate this list, replace missing real controls with simulation, lower
sample sizes, change prompts/priors/budgets/thresholds, or tune on held-out data.
An uncontended shared-server window would materially reduce the remaining time;
continue using exclusive flock around EACH request. Never start/stop/reconfigure
that server.

## 8. Handoff

Runner exited at a between-trial checkpoint; no in-flight client and no partial
measured trial remains. No stop marker remains in `snapshots/`. Resume the same
frozen runner without refreezing and append the existing log:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -u LEARNING/experiments/mission-050/v2/real_agent.py run >> LEARNING/experiments/mission-050/v2/real-run.log 2>&1
```

It skips all 393 completed trials, verifies design/source hashes, preserves the
frozen fixed-step-1 comparator, and proceeds to the listed next trial. Afterwards
run `verify_v2.py`, `report_results.py` and `finalize.py` as in `commands.md`.
Preserve this checkpoint's seal/reports/accounting beside updated versions before
regenerating derived outputs. Never delete a genuine partial snapshot to rerun.

Orchestrator: record candidate verdict, incomplete scope and ROI in the authorized
manifest/ledgers; run APM install when integrating guardian changes. The proposed
follow-up inquiry is in `operational-notes.md`; this worker did not write inbox or
mission files. Preserve both v1 and failed v2 evidence.
