# MISSION-041 v2 completion: recommendation FALSIFIED

1. Changed files

Added `LEARNING/experiments/mission-041/v2/`: frozen protocol, specifications,
20 references, generated claim/target adapters and suites, curation, scoring,
positive controls, raw evidence, audit, local investment assessment and this
report. Modified only the authorized plugin script `counterexample_eval.py`,
its `tests/test_counterexample_eval.py`, and `references/property-falsification.md`.
V1 evidence is intact. The manifest was already modified at intake and was not
edited by this worker. No git mutation, APM install, mission/index/backlog/ledger
write, server start, stop or configuration change occurred.

2. Summary

Executed a 20-component, five-family fixture experiment under a frozen v2
protocol. The existing-suite and authoring-time limitations below prevent a
claim that every interpretation of the full mission experiment is satisfied. Every original v1 component is rerun and numbers are not
pooled. All 120 candidates were curated before scoring: 47 non-equivalent,
73 equivalents (documented exhaustive or structural proofs), 0 unresolved.
Independent oracle verification covered 745,564 cases: 19 full finite domains,
plus reduced exhaustive + 10,000 sampled full-domain cases for v1 replay.

All three arms were executed. Property search detected 46/47, example generation
47/47; historical tests detected 12/47 overall, or **12/12 on their covered
mutants**. The fourteen new fixtures have no historical tests (35 mutants),
explicitly `NO_TESTS`. This absence is not a result about the quality of existing
project tests. A requirement for nonempty historical suites on all 20 would
remain unsatisfied; historical tests cannot be manufactured retroactively.

3. Exact experiment commands and observed outcomes

Commands ran from the stated codex-a worktree, Python 3.13.4, stdlib plus pytest.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider plugins/sanctuary-core/.apm/skills/evaluating-agent-work/tests/test_counterexample_eval.py
# 22 passed in 4.51s, including real descendant/pipe-holder timeout regression.
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider LEARNING/experiments/mission-041/v2/test_protocol.py
# 4 passed in 0.03s: distinct generators, independent oracle agreement,
# 20 components / five families, v1 reference preservation.
python3 -B LEARNING/experiments/mission-041/v2/prepare.py
# FROZEN 20 components, 40 suites, 64 cases each; 52 source/protocol hashes.
python3 -B LEARNING/experiments/mission-041/v2/curate.py > LEARNING/experiments/mission-041/v2/curation-output.txt 2>&1
# Exit 0: 745,564 oracle checks; 47 non-equivalent, 73 equivalent, 0 unresolved.
python3 -B LEARNING/experiments/mission-041/v2/run.py --development --output positive-control-results.json > LEARNING/experiments/mission-041/v2/positive-control-output.txt 2>&1
# Exit 0: 20/20 constant-return controls detected by both generated arms;
# historical tests 6/6 covered, 14 NO_TESTS. No errors/timeouts. Separate control.
python3 -B LEARNING/experiments/mission-041/v2/run.py --output primary-results.json > LEARNING/experiments/mission-041/v2/primary-output.txt 2>&1
# Exit 0: valid_measurement=true, 46/47 property, 47/47 examples,
# 12/47 historical (12/12 covered); errors=0, timeouts=0; FALSIFIED.
python3 -B LEARNING/experiments/mission-041/v2/audit_results.py > LEARNING/experiments/mission-041/v2/audit-output.txt 2>&1
# Exit 0: all_passed=true; counts independently recomputed; 105 recorded
# detection witnesses replayed; survivor's frozen 64-case suite replayed;
# freeze and curation seals verified unchanged. This is an audit, not a new run.
python3 -B LEARNING/experiments/mission-041/v2/write_report.py
# Reporting only; emits this report, investment assessment, evidence hashes.
```

The only server call was a read-only `urllib.request.urlopen` models probe with
an exclusive `fcntl.flock` held on `/tmp/sanctuary-mlx.lock` throughout the
request. HTTP 200 identified mlx-community/Qwen2.5-Coder-7B-Instruct-4bit;
`server-readiness.json` preserves the response. No inference or token telemetry.
Intake `git status --short` showed only the pre-existing manifest modification;
`git branch --show-current` reported feat/codex-a-mission-041.

4. Evidence paths (all relative to this v2 directory)

- `preregistration.md`, `freeze.json`: fixed contracts, generation instructions,
  thresholds, seed 41052, 64 cases, budgets and file hashes, frozen before curation.
- `specifications.py`, `references.py`, `manifest.json`, `suites.json`, claim and
  existing adapters: implementation-independent oracles, bounded domains,
  generation methods, historical provenance and exact generated case streams.
- `mutations.py`, `curation.json`, `scoring-lock.json`: six held-out operator
  families, all 120 classifications and proofs/witnesses, sealed before scoring.
- `primary-results.json`, `primary-output.txt`: complete observations, inputs,
  evaluation counts, statuses, budgets and measured durations.
- `positive-control-results.json`: separate 20 constant-return controls; do not
  interpret their generic summary's FALSIFIED label as the primary outcome.
- `audit.json`, `audit-output.txt`: verified raw counts, witnesses, survivors,
  denominator coverage and source/curation integrity.
- `investment.json`, `evidence-sha256.json`: prediction variance, runtime evidence,
  local assessment, and artifact seal; no fabricated token count or token ROI.

5. Thresholds, sample size, exact scope and verdict

| Registered threshold | Measured value and scope | Assessment |
|---|---|---|
| Detection >=80% | 46/47 = 97.872340% non-equivalent held-out mutants, 20 components | MET |
| Gain >=20pp over equal-budget examples | 46/47 minus 47/47 = -2.127660pp, same 47 mutants | MISSED by 22.127660pp |
| False alarms <=5% | 0/20 = 0% verified references, property arm, 64 frozen cases per reference | MET in this sample |
| <=10 min/component | max 10.883708s (path), n=20, total 25.879193s; authoring unmetered | MET computationally; cannot decide end-to-end |

Each component's time includes generation of both suites, independent reference
verification, equivalent curation, three reference arms and three mutant arms.
Excludes environment/setup tests, specification/constructor authoring, the
separate control run and postmeasurement audit. It does **not** measure end-to-end
agent authoring duration or external model generation. Both generated arms have
the same ceilings: 64 case slots, 1s generation, 2s isolated search per target.
They stop on first failure, so consumed work differs. The raw runner
summary labels the computational runtime flag true; it does not certify the
broader end-to-end manifest timing interpretation. All generation times were
under 1s. There were no candidate exceptions, evaluation errors or timeouts.

| Component | Non-equivalent n | Historical kills | Example kills | Property kills | Computational s |
|---|---:|---:|---:|---:|---:|
| lease | 1 | 1 | 1 | 1 | 0.422577 |
| renew | 1 | 0 | 1 | 1 | 0.307061 |
| release | 1 | 0 | 1 | 1 | 0.236568 |
| fence | 1 | 0 | 1 | 1 | 0.240098 |
| path | 3 | 3 | 3 | 3 | 10.883708 |
| relative | 4 | 0 | 4 | 4 | 0.666284 |
| extension | 4 | 0 | 4 | 3 | 0.595691 |
| path_key | 3 | 0 | 3 | 3 | 0.568632 |
| replay | 4 | 4 | 4 | 4 | 3.399025 |
| last_replay | 5 | 0 | 5 | 5 | 0.697980 |
| revision | 2 | 0 | 2 | 2 | 0.484722 |
| tombstone | 4 | 0 | 4 | 4 | 0.721475 |
| accounting | 1 | 1 | 1 | 1 | 0.941196 |
| clamp | 2 | 2 | 2 | 2 | 2.453439 |
| reserve | 2 | 0 | 2 | 2 | 0.745322 |
| refund | 1 | 0 | 1 | 1 | 0.404019 |
| schema | 1 | 1 | 1 | 1 | 0.468448 |
| migration | 2 | 0 | 2 | 2 | 0.432276 |
| required | 2 | 0 | 2 | 2 | 0.611606 |
| enum_map | 3 | 0 | 3 | 3 | 0.599066 |

Recommendation: **FALSIFIED for this frozen benchmark**, because the mandatory
+20pp improvement threshold was missed. The property arm was 1 kill worse than
the example baseline. This does not establish that claim-driven generation is
universally inferior. Even a broader interpretation that considers the existing
control or end-to-end runtime scope incomplete cannot justify CORROBORATED:
the independently observed required gain was absent.

6. Defects and uncertainties

- Property survivor: `extension_stable_sequence_sort`. The constructor covers
  empty paths and suffix categories but almost all generated paths have one
  token. It misses the interaction between reordering and selecting the final
  basename. The existing frozen suite was NOT repaired or replaced. Example
  case `['A.py', 'a.b.c', 'a.PY']` expects `PY`, while the mutant returns `c`.
  Independent pre-scoring witness `['A.py', '.env']` expects empty suffix,
  mutant returns `py`.
- Historical control is complete only on six v1 components. Its 25.53% overall
  kill rate is heavily confounded by 35 mutants with no historical tests. It
  achieves 100% on the 12 mutants it actually covers. It is the old exhaustive
  oracle-check recipe, not a typical hand-maintained production unit suite.
- Convenience fixtures and lossy-representation mutants, single author and seed;
  procedural operator holdout, no independent blinding. Equivalents include
  inapplicable identity transformations; excluded before scoring, never counted
  as surviving semantic mutants. Results do not generalize to all codebases.
- Independent means separately formulated specification versus implementation;
  the same author wrote both, so common misunderstandings remain possible.
- The six original domains are unchanged. Replay verification is not exhaustive
  over its immense full domain. Its equivalent exclusions are structural
  string-operation identities; all four non-equivalents have explicit witnesses.
- Zero alarms on 20 references is a sample rate, not proof of population <=5%.
  Even assuming independent reference sampling, the exact one-sided 95% upper
  bound for 0/20 is 13.9108%.
- The intervention uses fixed author-translated Python constructors, not a
  separately measured LLM or autonomous compiler; runtime excludes authoring.
  No measured model usage output was available; token economy remains unscored.
- The generic positive-control summarizer uses primary threshold labels and
  carries the held-out equivalent count; that output is diagnostic only.
- POSIX process-group kill covers ordinary descendants and inherited pipes,
  not processes deliberately escaping sessions, memory limits or hostile code.
- Guardian wakeup can reap a server, so its runtime side effect was bypassed.
  Control-plane scripts/database were absent. Official score/budget scripts,
  coordination checkin and APM refresh would write outside allowed paths.
  All are explicitly handed to the orchestrator, not run by this worker.

7. Remaining work

No additional primary scoring is required to establish the observed gain
threshold miss in this frozen experiment. Full historical-suite coverage and
prospectively metered authoring remain open under a stricter full-mission reading. Do not
rerun with changed constructors or pool v1 results. For a claim about historical
suite performance on 20 nonempty suites, a new prospectively selected benchmark
with real historical tests is required. For an end-to-end <=10-minute authoring
claim, prospective per-component authoring clocks or model generation telemetry
are required. Neither can be inferred from this computational timing result.

8. Handoff

Orchestrator: retain the negative gain and raw survivor. Record the scoped verdict
and caveats in MISSION-041 and parent MISSION-052, perform official measured usage
and quality bookkeeping only from your available telemetry, and refresh installed
APM copies after integrating the three plugin changes. Do not rerun the sealed
experiment or claim the fourteen missing historical suites were created.
No commit/push/merge/checkout was performed. Suggested new inquiry (for the
orchestrator's replenishment): independently curated real components with
historical tests, newly withheld operators, and compositional path/event
properties. The model server is unchanged; no model inference was needed.
