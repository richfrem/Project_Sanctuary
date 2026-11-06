# MISSION-041 v3 results (scored run; verdict FALSIFIED, scoped)

Freeze committed to git at `41f2a57b5` before scoring; `freeze_tools.py --verify` passes. Scoring: `run.py --phase2 --freeze-commit 41f2a57b52a46c0e6fd472e0453bee02e1e74737` (run by the orchestrator, output `phase2-results.json`).

| Threshold | Measured (n = 80 non-equivalent held-out source-level mutants, 20 components) | Met |
|---|---|---|
| (a) detection >= 80% | property 53/80 = 66.25% | no |
| (b) +20pp over equal-budget examples | examples 63/80 = 78.75%; gain -12.5pp (existing-tests arm 55/80 = 68.75%) | no |
| (c) false alarms <= 5% | 0/20 references | yes |
| (d) <= 10 min/component | max 1.2 s, computation only (authoring time not measured) | yes (conditional) |

Verdict: FALSIFIED under the manifest's literal criteria (any threshold missed falsifies).

## Scope and limits (from independent review by a Claude reviewer of a Codex delivery; accepted conditionally on these statements)
1. The property arm is "full independent oracle on 16 structured inputs", not a test of claims without an oracle: the metamorphic relations were redundant because every output was first checked against the full oracle (all 53 property detections have reason `independent_specification`). The result is a negative finding for this input-selection scheme. A no-oracle property arm (relations as the only check) is untested.
2. Some property misses are generator-coverage gaps unrelated to the hypothesis: `schema` (version 0 never exercised) and `required` (no duplicates generated). Fixing them would raise the property arm to about 72%, still short of 80% (reviewer estimate, not re-run).
3. The 80 mutants are 20 clusters of 4 near-clones, so the effective sample is closer to 20 components than 80 independent faults.
4. The hypothesis speaks of an agent translating claims into tests and of 10 minutes per component; what ran is hand-written deterministic generators and a compute-only runtime. The verdict is scoped to this benchmark and implementation, not to agent-authored properties in general.
5. Holdout is procedural, not blind: one author wrote pools and mutants; order rests on mtimes, a seal and a script assertion (the seal itself was not committed separately). The v3 freeze commit precedes scoring.
6. Superseded evidence: v2 (ceiling effect, no-op mutants, input-transform mutants) is retained with `v2/errata.md`; v1 was a 6-component pilot. Neither is pooled.
7. No token counts were recorded for the worker; measured usage is in the orchestrator record.

## Verification of record
`freeze_tools.py --verify` and `test_protocol.py::test_scoring_has_not_run` are PRE-scoring gates: they pass at the freeze commit `41f2a57b5` (and passed immediately before scoring) and fail by design once results and this note are added to the folder. To re-verify the freeze, check out `41f2a57b5` and run `freeze_tools.py --verify`. The experiment folder is not part of any collected test suite; the evaluating-agent-work suite passes with the harness changes (110 passed when run together with the protocol tests, plus this one expected gate failure).
