# MISSION-041 results: scoped hypothesis FALSIFIED

The primary preregistered held-out run detected **24/24 (100%)** semantic mutants
with property search and **22/24 (91.67%)** with the ordinary randomized example
baseline. The **+8.33 percentage-point** gain misses the required +20pp. Any missed
threshold falsifies this scoped hypothesis. Do not mark MISSION-041 corroborated.

| Hypothesis threshold | Observed | Assessment |
| --- | --- | --- |
| Detection >=80% | 24/24 = 100% | Met |
| Gain >=20pp over equal-budget examples | 100% - 91.67% = +8.33pp | **Missed by 11.67pp** |
| Reference false alarms <=5% | 0/6 = 0% | Met in this small sample |
| Runtime <=10 minutes/component | maximum 0.872281s; total 5.087623s | Met |

| Component | Property detections | Example detections | Aggregate runtime |
| --- | --- | --- | --- |
| lease | 4/4 | 3/4 | 0.833s |
| path | 4/4 | 4/4 | 0.860s |
| replay | 4/4 | 4/4 | 0.837s |
| accounting | 4/4 | 3/4 | 0.835s |
| schema | 4/4 | 4/4 | 0.850s |
| clamp | 4/4 | 4/4 | 0.872s |

All 24 mutant witnesses distinguished mutant/reference against independent
specification oracles. No equivalents were silently dropped. Both methods
accepted all six references; no errors or timeouts occurred in scored runs.
The baseline misses were `lease_strict_expiry` and `accounting_strict_limit`.
Every search had seed 41, a 64-evaluation ceiling, and a 2-second wall ceiling;
all work for each component shared a 600-second aggregate ceiling. Early failure
stops a search, so budgets are equal ceilings, not equal actual CPU expenditure.
These results measure input generation strategies with the same specification
information; this is not evidence of an autonomous natural-language compiler.

## Tests and experiments

All commands ran offline from the worker worktree using Python 3.13.4.

1. `python3 -B LEARNING/experiments/mission-041/build_fixtures.py` generated frozen
   fixture sources before scoring. No fixture/generator changes after scoring.
2. `python3 -B LEARNING/experiments/mission-041/verify_references.py`: all
   731,817 oracle cross-checks passed
   in 0.582s. Five stated domains exhaustively covered; replay
   covered 55,987 reduced-domain sequences plus 10,000 seeded full-domain samples.
3. Development benchmark (command below with `--split development`): six
   constant-return development faults, 6/6 killed by both methods, zero alarms;
   delta 0pp, thus scoped thresholds falsified. This is harness validation, not
   part of the held-out numerator/denominator.
4. Primary held-out benchmark (command below): results above. Raw per-search
   inputs, statuses, evaluations, budgets, timings, and SHA-256 source hashes in
   `heldout-results.json`.
5. Exact same held-out command repeated into `reproduction-results.json`:
   identical semantic report after removing elapsed times, including discovered
   inputs and evaluation counts. `reproducibility.json` also confirms every
   frozen fixture and preregistration file is unchanged.
6. `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest plugins/sanctuary-core/.apm/skills/evaluating-agent-work/tests/test_counterexample_eval.py`:
   **21 passed in 2.36s** on the final run (plugin autoload also disabled). The
   earlier 18-test version passed in 2.49s. Tests include real hanging code killed
   as TIMEOUT, deterministic replay, oracle exceptions/import errors, candidate
   exceptions, input mutation isolation, equal budgets, exact metric thresholds,
   false-alarm counting, aggregate deadlines, and invalid/empty measurements.
   Final output is retained in `pytest-output.txt`.

Primary benchmark command:

```bash
python3 -B plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py benchmark \
  --manifest LEARNING/experiments/mission-041/fixtures/manifest.json \
  --split heldout --seed 41 --trials 64 --seconds 2 --component-seconds 600 \
  --output LEARNING/experiments/mission-041/heldout-results.json
```

## Limits and handoff

This is a six-component worker delivery, below the manifest's full 20-component
experiment. The held-out operators are disjoint from the development
constant-return operator and live in separate files, but the same author wrote
both splits, witnesses, and properties: this was not independently blinded.
References have independent formulations, but replay is not exhaustively
verified over its full stated domain. Zero alarms on six references is a sample
observation, not a population guarantee of <=5%. One primary seed and one exact
reproduction do not establish robustness across seeds or domains. No existing
project-test baseline or external model dispatch was measured.

Deadline enforcement isolates a trusted worker process, not hostile code or
its descendants; memory is not bounded. Timing is environment-dependent.
No counterexample within a budget and no timeout ever proves correctness.

Investment prediction was recorded in `preregistration.md` before scoring:
detection/alarm/runtime expectations met; anticipated insufficient baseline
advantage occurred. The reusable harness, explicit misses, and raw artifacts
are the return. Local quality assessment: useful outcome 2, learning depth 2,
verification rigor 2, anti-Asch integrity 3; token economy unscored (no usage
telemetry and no self-reported token counts). No official scoring/budget ledger
was changed; supervisor should use measured orchestrator telemetry.

Supervisor: retain the falsified result, integrate allowed files, refresh APM
installed copies with `apm install` at the repository root, and perform any
mission/coordination/ledger updates separately. Those writes were outside this
worker's authorization. The pre-existing MISSION-041.json modification was
left untouched. Do not tune these held-out cases and re-label them unseen.
A useful next inquiry is an independently curated larger benchmark testing
whether boundary-guided generation adds >=20pp against stronger baselines.
No git mutation or network operation was run. Session ID unavailable.
