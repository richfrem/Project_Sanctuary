# MISSION-041 scoped preregistration

Frozen before the first development or held-out benchmark run. Seed 41, 64
candidate evaluations per method per target, 2 seconds per search including
subprocess startup, 600 seconds total per component including curation,
references, and both methods. One scored seed; no tuning after held-out scoring.
A second seed may be reported as sensitivity, never replace the primary seed.

Prediction: property detection >=80%, reference alarms 0%, maximum component
runtime <600s. Improvement over the ordinary randomized example baseline may
miss +20pp: many small semantic mutants are easy for either method. Any missed
threshold falsifies the scoped numerical hypothesis. No token estimates.

Six components cover leases, absolute root-clamped paths, first-event replay,
bounded accounting, schema transitions, and clamping. Each has four held-out
semantic mutants plus a development constant-return fault. Explicit witnesses
prove non-equivalence against specification oracles. Curation witnesses are never
fed to either scored method. Development uses only the constant-return operator;
held-out operator families are not used to change the harness or generators.

The example baseline generates uniform bounded-domain inputs (not a repeated,
handpicked list) and compares outputs with independently expressed expected
values. Property search uses a claim-informed boundary-biased generator for
lease/accounting/clamp, and the same uniform generator for the other components.
Both methods have the same oracle information, seed, maximum trials and wall
budget; they stop at their first counterexample. This measures generator choice,
not an automatic natural-language-to-property compiler or external-agent skill.
Equal budgets are ceilings; actual work differs after early stopping.

Oracles: disjunctive lease predicate; stdlib posixpath absolute normalization;
reverse dictionary replay model; enumerated admissible charge amounts; exhaustive
3x3 transition table; ordered-statistic clamp model. References are reviewed and
will be checked against these independent formulations over bounded exhaustive
or deterministic coverage. False alarms count reference implementations rejected
by property search / all six references. Also retain baseline reference results.
ERROR or TIMEOUT invalidates an otherwise passing measurement; neither is a kill,
a survivor proof, or a silent denominator exclusion. All declared held-out mutants
stay in the scoring denominator; a failed witness invalidates the run.

Limitations fixed in advance: this worker has six components, whereas the mission
manifest requests 20. One author wrote specifications, generators, and both
fixture splits; separation is procedural and file-based, not blinded. Mutants and
witnesses were authored together. No pre-existing project test baseline exists
for these new toy components. No claim of full mission corroboration is justified
even if the scoped numerical gates pass. Reference alarm denominator is small.

Investment: a reusable offline harness plus measured counterexamples should be
more useful than a narrative-only evaluation. Runtime and test outcomes are the
local expenditure/effectiveness evidence; token metering and official contribution
scoring are reserved for the supervising worker, with no ledger writes here.
