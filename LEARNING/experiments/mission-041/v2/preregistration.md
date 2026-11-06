# MISSION-041 v2: frozen 20-component protocol

Authority: `LEARNING/missions/MISSION-041.json`, read 2026-11-06 UTC in the
codex-a worktree. The user authorizes this experiment, forbids git mutations and
mission/index/backlog/ledger writes, and limits writes to the experiment and
three named evaluating-agent-work files. No subagent or external model dispatch.
The primary outcome is one frozen seed, 41052. No tuning after curation or scoring.

Prediction before measurements: property detection >=80%; zero reference alarms;
all computational component times <600s; improvement over the strong example
baseline probably <20pp. Any missed registered threshold recommends FALSIFIED.
A useful return is a reproducible three-arm comparison and equivalence evidence,
including a negative finding rather than an exaggerated property advantage.

## Components and independent authority

Twenty *distinct* bounded Python contracts, four in each family:

| Family | Components |
|---|---|
| Lease decisions | acquire (`lease`), renewal, release, fencing |
| Path normalization | root-clamped absolute (`path`), relative, suffix extraction, case-sensitive canonical key |
| Event replay | first sum (`replay`), latest value, monotonic revision, tombstones |
| Accounting | bounded charge (`accounting`), clamp, atomic reservation, capped refund |
| Schema transitions | state machine (`schema`), versioned key migration, required fields, enum conversion |

`specifications.CONTRACTS`, `domain`, and `expected` are the normative executable
specifications. `references.py` independently implements those contracts using
imperative algorithms; it never imports the specification or its state table.
Six original v1 reference functions are behaviorally preserved and **rerun**;
new protocol, new suites and new operators mean no pooling of v1 measurements.
All six v1 domains remain unchanged. `extension` input is a sequence of zero to
three names from its explicit alphabet. New domains are exactly enumerated by
`domain`. The v1 replay domain is too large to exhaust: retain v1's reduced
exhaustive domain plus 10,000 seeded full-domain samples. Its equivalence
exclusions require a structural full-domain proof, not sampled equality.

No generated test may use the implementation as oracle. Property checks and
example expected values share the same independent specification information;
the measured intervention is claim-specific input construction. `property_generate`
has twenty separate constructors and cannot call or alias `example_generate`.
The example generator draws ordinary independent bounded random examples.
The claim translation is performed by the author into deterministic Python,
not by a separately evaluated LLM, automatic compiler, or model comparison.
No stochastic model-generation effect is claimed.

## Frozen generation instructions (no model requests)

The following are the complete authoring instructions, instantiated for each
`CONTRACTS[name]`, with all resulting executable constructors frozen before
measurement. There are no hidden server prompts or post-result repair prompts.

Example arm: "Given this independent bounded contract and its legal input domain,
construct 64 ordinary randomized examples using independent uniform choices for
fields, length and alphabet. Do not use mutations, curation witnesses or candidate
outputs to select examples. Compare outputs to the independent specification."

Property arm: "Given this independent bounded contract, derive its important
partitions, boundary relations, collision/ordering situations or transition
categories. Construct 64 legal cases covering those claim-specific situations,
with seed 41052. Use a separate constructor, never the example generator, candidate
implementation or mutation catalog. Check the independent specification."

Equal budgets for the two generated arms: 64 cases; 1s suite-construction ceiling;
2s isolated search ceiling per target including worker startup/import; stop on
first failure. Equal ceilings, not forced equal consumed CPU or tokens. Suite
creation durations and immutable case lists are saved separately in suites.json.
No observed held-out case may be added. No retries, replacement seed, additional
scored runs, model calls, or optimization of this frozen protocol.

## Existing tests: provenance and missing coverage

A genuine historical control is the v1 `verify_references.py` test recipe and
unchanged `fixtures/claim_<name>.py` check functions for the original six.
The existing arm replays those exact domain streams and oracles, not v2's new
generated cases: 2,178 lease; 488,281 path; 65,987 replay; 35,937 accounting;
139,425 clamp; 9 schema cases. Existing exhaustive controls get a 30s process
ceiling because they are not the equal-budget generation comparison.

There were **no existing tests** for the fourteen newly authored fixtures at
intake. Record `NO_TESTS`, tested=0, on those targets, with zero observed kills.
Do not fabricate a historical suite. Report both overall mutant denominator and
covered-component denominator. This is an explicit limitation of the existing
control, not of the example/property numerical comparison. No inference that
existing tests generally have zero detection is permitted.

## Operator-level holdout and pre-scoring equivalence curation

Development operator: constant-return None only. Existing unit tests and v1
constant-return development controls validate infrastructure. No held-out operator
is used to tune the generators or harness. Held-out families are distinct lossy
input-representation operators, rather than v1 source-level condition changes:
unsigned 2-bit conversion, casefold identifiers, stable sequence sorting, tail
loss, signedness erasure, one-character identifier encoding. Each recipe is
applied to each of 20 references: **120 candidate mutants**, all retained in
curation. Array shape is preserved for fixed-arity arguments; nested request/key
sequences get explicit sequence transforms. Lossy numeric inputs are repaired
only to preserve ordered clamp bounds and subordinate refund balances.

All sources, suites, rules, seeds, limits and this document are SHA-256 frozen by
prepare.py BEFORE curate.py executes any held-out candidate. Curation first
cross-checks reference vs independent oracle. For every operator/component it
records the first valid distinguishing witness, or proves equivalence over the
entire finite declared domain. For replay, string-only transformations are
structural identities because no legal replay case contains a string. Other
unwitnessed replay mutations are UNRESOLVED, not silently excluded. Curation never
modifies generator input streams. Witnesses are not fed to scored search arms.

Equivalent exclusions and witnesses are written to curation.json, then sealed by
scoring-lock.json BEFORE any arm scores a held-out candidate. Non-equivalent
mutants all remain in the scored denominator, regardless of detected/surviving.
A component with zero non-equivalent mutants, a bad oracle, or unresolved curation
prevents a full valid measurement. Holdout is procedural and file-based, written
by one author; no independent blinding or population sampling is claimed.

## Measurements and decision rules

Primary detection = property kills / all curated non-equivalent held-out mutants.
Gain = that rate minus example kills / the identical mutant set, in percentage
points. Property reference false alarms = rejected references / 20 distinct
verified references (not / generated case count). Also report each other arm's
reference alarms and absent-test count. No detection for TIMEOUT, ERROR, NO_TESTS.
Infrastructure ERROR or TIMEOUT invalidates measurement instead of becoming a
silent exclusion. Candidate exceptions on valid input would count as witnessed
failure; preserve reason and case. Finite exhaustion is not correctness proof.

Thresholds: (a) >=80% detection, (b) >=20pp gain over equal-budget examples,
(c) <=5% reference alarms, (d) <=600s per component. Any miss falsifies the
numerical hypothesis for this benchmark. Insufficient/invalid observations mean
cannot decide. No confidence-bound requirement is added: nevertheless report
that n=20 and convenience mutants do not establish population alarm guarantees.

Component computational time includes both suite constructions, independent
reference verification, equivalence curation, all three reference controls, and
all three arms on all scored mutants. The clock excludes initial environment,
authoring/specification design and setup tests; this is a runtime measurement,
not a measured claim about end-to-end authoring labor. Report that scope with
the threshold. Runtime and process time are measured, tokens only if emitted by
a tool/server. No self-reported token numbers. No token-count telemetry is
available for the authoring session; do not invent a count or token ROI ratio.

Checkpoint after each component, preserve all earlier artifacts. One optional
constant-return positive-control run is permitted, reported separately, never
pooled with held-out numbers and never used to change frozen generators. Verify
freeze and curation hashes before and after scoring. Do not overwrite outputs.

## Operational constraints and handoff

Use Python stdlib, pytest for checks. Timeout starts each worker in a new POSIX
session, SIGKILLs its process group and reaps its direct child; this is containment
of ordinary descendants, not a hostile-code sandbox. No server start/stop/config.
This offline experiment requires no server request; any future request must
hold exclusive fcntl.flock on /tmp/sanctuary-mlx.lock throughout the request.

The generic control-plane entrypoints and database are absent in this checkout.
Its registration paths are also outside worker authorization. The guardian
wakeup's runtime probe can call reap_if_idle; skip that side effect to honor the
shared human-managed server rule. Read-only orientation was performed instead.
Ledger-writing budget/scoring scripts and APM installation are outside allowed
paths; leave official evaluation, install refresh and verdict registration to
the orchestrator. Save an evidence-local investment assessment and a next inquiry
suggestion without writing any mission or ledger.
