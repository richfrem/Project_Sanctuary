# MISSION-041 v3 preregistration — phase 1 only

Authority: user instruction to codex-a, continuing MISSION-041. Preserve v1/v2,
no git commands, no manifest/ledger/index writes, no external model dispatch,
no scoring in phase 1. Allowed writes are this experiment directory and the
three evaluating-agent-work files named by the mission. `v2/errata.md` records
the independent review; older results are not pooled or rewritten.

## Question, prediction and decision

Test the original numerical hypothesis on **this bounded convenience benchmark**:
property detection >=80% of curated non-equivalent held-out faults; advantage
>=20 percentage points against equal-budget examples; <=5% property reference
alarms; <=600 seconds computational time per component. Any missed threshold
falsifies the numerical hypothesis for these fixtures. Infrastructure errors,
timeouts or missing components make the measurement incomplete/INVALID and must
not be counted as detections or silently excluded. Exhaustion is not correctness.

Prediction made before phase-2 measurements: reference checks should pass; sparse
faults should prevent an aggregate example ceiling; a >=20pp property advantage
is uncertain and might fail. A negative result is useful. No claim about automatic
natural-language compilation, comparative model quality, production fault
frequencies or end-to-end authoring time is tested. No token count is available
for this session; do not invent one or an ROI ratio. `preparation-timing.json`
records actual computational preparation telemetry. Official budget/contribution
ledger writes are reserved for the orchestrator; no contribution scoring is run.

## Twenty independent finite contracts

`specifications.py` is the normative specification and full finite domain.
`components.py` separately implements each contract without importing it.
Curation checks every reference input against the independent specification.

| Family | Components | V3 domain change / claim |
|---|---|---|
| Lease | lease, renew, release, fence | clocks 0..127 or 0..31; reentrant release count; fencing tokens 0..255 |
| Path | path, relative, extension, path_key | sixteen-token alphabets, all sequence lengths 0..4; absolute/relative parent handling, suffix and case-sensitive keys |
| Replay | replay, last_replay, revision, tombstone | first/last event semantics, monotone revision and delete/put state; all bounded sequences |
| Accounting | accounting, clamp, reserve, refund | guarded charge, ordered bounds, atomic reservation and capped refund |
| Schema | schema, migration, required, enum_map | version-gated transitions, set-valued key renaming, required fields, case-sensitive enum conversion |

These are separate contracts, not aliases or renamed replicas. They are synthetic
small programs, not harvested production components. V3 changes bounds and some
contracts (reentrant release, version-gated transitions); therefore no numerical
comparison of v3 versus v2 detection is an intervention estimate. Domain sizes
range from 2,304 semantic inputs (schema) to 139,425 (clamp). All 20 domains are
exhausted; no sampled equality exclusions. Migration's set representation and
order-independence argument are explicit in `representation-proof.md`.

## Operator separation and temporal seals

The first v3 artifact was `operator-split.json`, written BEFORE specifications,
generators, implementations and mutants. Development families are constant
return / unconditional exception. Held-out source-edit families are boundary
shift, branch deletion, wrong branch value, wrong state update. The development
families are disjoint from the conditional edits used in the held-out catalog.
Generic infrastructure unit fixtures are not a development benchmark score.

Next `specifications.py`, `pools.py`, and `suites.json` were sealed by
`seal_pools.py` into `pool-seal.json`. Only THEN was `build_mutants.py` authored
and its catalog materialized. No property or example generator, literal unit
suite, frozen case or normative specification changed after that seal, including
after kill densities were inspected. Property construction uses only claims and
domains: equality/adjacency, ownership, normalization neutrality, duplicate
replay, version identity, monotonicity, set-order and case laws. No mutation
recipes, witnesses or candidate outputs feed construction. Family names were
known to the author; this is **procedural withholding, not independent blinding**.
The pre-mutant seal was not committed separately and does not independently
prove wall-clock authorship order; the orchestrator's phase-1 commit anchors the
final freeze only. No hidden author or model holdout is claimed.

Mutations are real source edits in materialized functions in `mutants/*.py`.
They do not call a reference or transform reference inputs. Source edit recipes,
old/new spans, operator and file paths are retained in `mutants.json`. Every
recipe matches once, changes the parsed AST, and compiles. The benchmark uses
85 applicable candidates: **80 non-equivalents, 5 true equivalents, 0 syntactic
no-ops and 0 inapplicable recipes**. Every component contributes exactly four
non-equivalents; multiple recipes within a component are related, not independent
randomly sampled production faults. Five inclusive/exclusive comparisons change
branch execution at equality but preserve the output; full-domain comparisons
prove these true equivalents. None is excluded merely for lacking a witness.

## Curation and diagnostic density (not arm scoring)

`curate.py` compares every candidate against the independent, reference-verified
specification over the COMPLETE declared domain, retaining distinguishing counts,
first witness, domain size and classification. It does not import pools or read
suites. `curation.json` contains 4,303,806 candidate/input comparisons; every row
is shown in `kill-density.md`. Exceptions on legal inputs count as differences.
No candidates were replaced, thinned or excluded after density inspection.

Non-equivalent density: minimum 0.043403%, median 3.065488%, maximum 22.407553%
(the intended approximately-20% high end). Fifteen mutants are below 1%; 44 are
at most 5%; ten exceed 20% (all are 22.407553%). Dense mutants may individually
approach a random ceiling, but the sparse majority makes aggregate saturation
implausible under the frozen budget. For orientation only, 16 independent uniform
draws against a 1%-density fault have detection probability 1-(.99)^16, about
14.9%; an equality boundary supplied as a plain example can of course detect it.
This density argument is not a measured baseline result. Actual generated-arm
detection is unknown until phase 2. Curation witnesses are never added to suites.

## Arms, generators and equal budgets

All case lists are frozen in `suites.json`, seed **41053 + component index**.

* Property: eight claim-specific two-call probes, **16 candidate calls total**.
  Every output is checked against the independent specification and the paired
  outputs also check a metamorphic law (equality, reversal, monotonicity).
  Neither side is free: early failure counts calls already made. Generators are
  twenty claim-specific constructors; no calls to the example generator.
* Example: **16 candidate calls total**: four conventional literal examples from
  the engineer suite, then twelve independent uniformly sampled semantic domain
  inputs, with replacement. The first examples include ordinary empty, zero,
  equality, out-of-order, parent, duplicate or cap cases as appropriate. They are
  not mutation witnesses. Expected values come from the same independent contract.
  This is random sampling plus plain engineer examples, not an adversarially weak
  baseline. The complete exact list, including all boundaries, is reviewable.
* Existing-style unit tests: five or six conventional examples per component,
  written as literal input/expected pairs BEFORE mutant recipes were authored.
  All twenty are **newly authored v3 suites**, not claimed historical coverage.
  They include normal, rejected and boundary cases; they are not exhaustive.
  This arm is contextual and has its natural suite size, not a claim of equal
  generation budget. V2's historical coverage limitation remains in its errata.

Two generated arms get identical 2-second worker ceilings per target, including
startup/imports; no adaptive generation, retries, extra seeds, shrinking, tuning
or positive-control benchmark runs. Sources are deterministic Python, not model
prompts. `harness.execute` charges every call, deep-copies input and distinguishes
target exceptions (counterexamples) from oracle/import failures (ERROR). Workers
are disposable subprocesses; POSIX timeouts kill process groups. Non-POSIX kills
the direct child only. This is trusted-code containment, not a malicious sandbox.
No server, MLX request or network dependency exists.

## Future scoring and runtime accounting

Property/example/existing detection uses the IDENTICAL denominator of all 80
non-equivalent held-out candidates; equivalents remain visible in curation only.
Report per-mutant outcomes and per-component outcomes before aggregate rates.
Report reference alarms for each arm / all 20 references, including the property
rate separately. No held-out arm has run during phase 1. Phase-1 validation runs
only reference suites and ordinary harness unit fixtures, plus exhaustive curation.

Component time in phase 2 includes measured curation/reference enumeration time,
all three reference-arm executions, all three arms on its non-equivalents, startup,
and conservative suite generation overhead. The whole measured 20-component
suite rebuild duration is charged to EACH component rather than inventing a
per-component allocation. Authoring, documentation, generic harness unit tests,
initial setup and git review/commit are excluded; report that limitation beside
the 600-second threshold. No end-to-end labor-time claim follows.

`run.py` is supplied for phase 2 but is NOT invoked now. It checks frozen inventory
and hashes before/after execution, requires explicit `--phase2` and a full
`--freeze-commit` attestation from the orchestrator, refuses to overwrite a prior
result/checkpoint, retains all failures, and enforces the aggregate runtime cap.
It does not verify git history itself and does not execute git. An interrupted
run requires orchestrator review; do not delete its checkpoint for a silent retry.
Frozen tests assert phase-2 outputs are absent, so they describe this phase only.

## Freeze, scope and handoff

`freeze_tools.py --seal` hashes EVERY v3 file except freeze.json itself and cache
files, plus v2/errata and the three modified evaluating-agent-work files. No caches
are expected because validation disables bytecode and pytest caching. Only
`phase2-results.json` / `phase2-checkpoint.json` may subsequently be added. The
freeze cannot hash itself: the orchestrator must commit it before phase 2 and
supply that commit hash. Do not modify frozen files after that commit.

The control-plane status script/database are absent in this checkout; read-only
skill and context orientation was performed. Guardian wakeup is not run because
its side effects can touch paths and the shared server outside this authorization.
No git commands, manifests, leases, ledgers, model-server actions, installed skill
copies, APM installs or soul persistence writes are performed. The source-level
platform guard is frozen, but installed copies must be refreshed by the
orchestrator during integration, outside this worker's path scope.

Next inquiry suggestion, without creating a mission: independently blind
operator authoring across vendors, with a second author deriving property pools
from contracts only, and test whether any advantage survives that separation.
