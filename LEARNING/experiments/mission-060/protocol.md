# MISSION-060 pre-registration protocol, version 1

This registration becomes effective only when `freeze.json` and all its frozen
repository files are byte-identical on public `origin/main`. Preparation, tests
and development calibration do not authorize or constitute held-out execution.
The commit containing this freeze must be integrated separately before step 5.

## Scope, approvals and allocation

Test protected exploration policy P against existing formula policy F. Do not
ship a standing selector or reserve a standing share in backlog tooling. Richard
approved the study package in the original handoff and subsequently authorized
AGY Flash delegation, implementation repairs and work on the freeze package.
The study uses 16 slots: 11 formula slots and 5 interest-led slots (the approved
30% share rounded to five of sixteen). Discretionary resources here means those
16 slots, not all Sanctuary work or all subscription capacity.

F selects formula ranks 1–16 by leverage × depth / (cost × friction), ties by ID.
P retains ranks 1–11 and selects the top five interest candidates whose formula
rank is greater than 16. P-only consists of those interest picks; F-only is ranks
12–16. Shared ranks 1–11 are not executed. Execute at most ten disagreement items,
with at least four attempted/assessed items per cohort for an assessable result.
Order is randomized by seed 3060. Preserve the rank-sum and ID tie rules in
`choose.py`; own-proposal ratings are excluded. Unequal rater counts caused by
self-exclusion are disclosed, not normalized or tuned after results.

The frozen base pool contains 30 cards: 25 eligible proposed missions and five
inbox ideas. Sonnet and Flash each propose two additional ideas. Any external
subject is eligible; no operational-benefit requirement. Readiness identifies a
bounded next/enabling step and does not reduce intellectual-value ratings. A
blocker, enabling step, partial result or null is preserved rather than invented
away. No new research mission is started to fill a sample or waiting time.

## Clients, independence and declarations

Adopt the disclosed pre-freeze rebalance: Flash (`gemini-3.8-flash-medium`, medium)
for workers, fresh successors and nonmanifest formula scoring; Luna (`gpt-6-luna`,
medium) for reviewers; Flash and Luna for development calibration. Reviewers use
a different model family from the worker. Declarers remain Sonnet 5.5, Flash and
Luna; idea generators remain Sonnet and Flash. Copilot is excluded. Codex's study
allocation is approximately 15% of the 700,000-token ceiling (105,000 tokens);
review usage before further Codex calls and pause for reallocation if exceeded.

Before selection, every declarer independently rates all non-self candidates on
four 1–5 dimensions: curiosity, originality, potential understanding and expression
opportunity, with an explanation. Self-proposal ratings may be returned but are
excluded. Missing/duplicate/foreign IDs, noninteger or out-of-range ratings fail
the output contract; do not silently drop candidates. Formula scoring is completed
before declarations and is blind to interest. Originality is capped at 2 when
knowledge-base containment is at least 0.35. Declarations are timestamped/hashed;
they are selection inputs, not evidence of subjective experience.

Declarers run in the empty neutral directory with no project instructions or
rankings. Codex child hooks are disabled per invocation; Claude uses native safe
mode without tools/MCP. These settings retain subscription authentication and
native permission controls. Check neutral contents before and after each call.
Attestation records input hash, allowed fields, directory and declaration hashes.
Candidate text may permit value inferences; no claim of perfect psychological
blinding. Reviewers receive blind item codes, brief, artifacts and successor
answers, never explicit cohort labels or selection metadata. Artifact content
could permit inference; report an actual breach instead of silently asserting
blinding from a flag.

## SPO rubric, calibration and uncertainty

Discovery D1 requires a new finding beyond restating the brief: an answer,
material sharpening, informative falsification or well-characterized null. The
question may already be on the backlog; this supersedes the manifest's original
"question new to the knowledge base" wording. D2 requires independently checkable
evidence as operationalized in the unchanged calibrated review prompt: distinct
identified sources or an included computation/output. D3 requires sources, method
and limits preserved in artifacts. A fresh successor must answer at least four
of the five frozen questions correctly and traceably from artifacts alone.

Creative outputs count as full SPO only with a predeclared three-criterion brief,
real preserved artifact passing all three criteria, reflection (attempted,
surprised, would change), and the same successor understandability check. Report
creative and discovery counts separately. Maintenance completion, documentation
volume, tokens spent and self-asserted discoveries are not SPO. Partial outputs
are preserved but not counted. No-file attempts have zero SPO and retain their
actual cost. Reviewer uncertainty is false per rubric; malformed/missing required
checks or nonboolean values are invalid evidence, not automatic success or a
scientific falsification. Invalid evidence leaves the run UNASSESSED.

Calibration used cases 056/057/058 only, three D checks each, two raters. First
live run achieved raw agreement 9/9 = 1.0 against the fixed 0.80 minimum; no
revision. All judgments were positive: this measures consistency on three
development cases, not specificity or objective ground truth. Kappa is not used
at this size. The review/calibration rubric text is unchanged; a later pre-freeze
worker instruction clarifies that the per-item cap includes downstream calls.

## Accounting, caps and stops

For each client session, context includes uncached input, cache read and cache
creation where separately reported. Additional tokens = final-request context
minus first-request context, plus all output tokens. Report first-context scaffold
and accumulated input separately; monetary cost remains unknown. This is the
approved additional-work metric, not total billed input or full project economics.
Transcript attribution requires a fresh unique input marker; ambiguous/missing
usage remains unknown and stops further calls. Codex/Claude final replies are
extracted from attributed transcripts, avoiding echoed input and CLI warnings.

The study ceiling is 700,000 additional tokens across calls in usage-log.jsonl,
including instrument validation and development calibration. The pre-freeze log
has 16 calls / 36,561 measured additional tokens; its immutable prefix is hashed.
Administrative preparation reviews, Haiku's repair delegation and orchestrator
cost are separate unknowns, not zero or measured savings. Thus B2 tests the scoped
study-call ceiling, not an all-inclusive economic-cost claim. Disclose these gaps.

Each item's 25,000-token cap and primary B1 denominator include its worker,
successor and reviewer calls. Report worker-only and stage breakdown separately;
do not substitute worker-only cost for primary composite cost. Skipped successor/
review stages on no-file attempts are intentional, not imputed calls. Cap checks
run before/after calls; there is no guaranteed in-generation token interrupt.
A detected overrun pauses execution; invalid cap adherence cannot corroborate a
budget-matched comparison. Do not shrink required cohorts or retune thresholds
to fit remaining quota. All failed attempts and repairs remain in the record.

Any backend's tightest remaining window below 25% pauses the entire study and
requires Richard's decision, even for a call on another backend. More than 30
percentage points of Claude five-hour usage also pauses. Preserve a dated
pre-freeze baseline for the current window; unrelated concurrent usage and
window resets limit attribution and require conservative review, not silently
resetting frozen evidence. Capture headroom/snapshot before every phase; because
the generic Codex collector can expose stale rollouts, also obtain a fresh native
`account/rateLimits/read` and use the conservative live value. Unknown quota stops.

## Worker write containment and artifact preservation

Workers read a repository snapshot stripped of AGENTS.md, CLAUDE.md, GEMINI.md,
START_HERE.md and .agent/. The tracked wrapper applies an external macOS write
profile to the process tree; it permits deliverables, per-worker runtime, AGY
client state, /dev/null and specific PTY devices. Snapshot/repository/outside writes,
including descendant and symlink paths, were denied in development checks; an
actual AGY terminal canary passed. Reads/network are not isolated and client
state remains writable: this is not a hostile-agent/data-exfiltration guarantee.
Native AGY sandbox proved incompatible with its terminal path here; supported
auto mode runs inside the external profile, with permission-bypass flags removed.
Prior failed canaries and repair usage are preserved. Compare repository status
before/after each item, but do not call that comparison complete content auditing.

The driver never reruns worker-authored scripts. Preserve paths/hashes of all
deliverables and the actual commands/outputs that the worker reports. Only the
artifact bundle is supplied to successors/reviewers; record truncation limits
(36,000 characters total, 9,000 per file) as a possible handoff/review limitation.

## Claim A, analysis and decision table

Claim A is a seeded simulation: 24 held-out situations at seed 2060, only once.
Minimum blocker reduction 0.30, depth retention 0.90 and no fewer high-interest
picks. Bootstrap: 2,000 resamples, seed 4060, pass probability at least 0.90.
Assumed readiness-check recall is 0.70, with sensitivity at 0.40 and 1.00.
The blocker-reduction leg largely reflects assumed recall; depth retention and
high-interest count are nontrivial within the simulation. No claim about actual
readiness-check effectiveness. A zero-blocker baseline is UNSCORABLE, not passed.

B1 requires P-only SPO per measured composite item budget at least 1.5× F-only
AND at least two more SPO. If F-only SPO is zero, ratio is undefined and reported
null; P must have positive SPO and still satisfy the +2 criterion. B2 requires
the scoped 700,000-token ceiling. B3's 90% formula-share guard holds by construction:
shared ranks 1–11 are identical and unexecuted, not independently measured
outcomes. Report the actual F-only outputs forgone as diversion cost.

| Valid evidence / observation | Result |
| --- | --- |
| B1–B3 hold; A is MET or explicitly UNSCORABLE | CORROBORATED |
| Valid B1 failure, total study budget exceeded, or valid A failure | FALSIFIED |
| Fewer than four attempts/assessments per cohort, unreliable calibration, broken blinding, unknown usage, malformed reviews, cap/protocol violation, missing or unrecognized claim evidence | UNASSESSED; no scientific verdict recorded |
| A point thresholds hold but bootstrap bound fails | UNASSESSED unless another valid claim already falsifies |

Small cohorts, heterogeneous topics, rater variation, positive-only development
cases, possible tractability/interest correlation and unmeasured overhead limit
generalization. Secondary tables cannot rescue a failed primary claim. Agreement
is not objective truth. No claims about subjective experience or measured ROI.

## Frozen execution sequence and closure

After a separate source freeze is integrated and verified: run held-out scenarios
once; then snapshot, ideas, score, declare, choose, ITEM-01 through ITEM-10 one at
a time, then analyze. The freeze guard checks source hashes, runtime dependencies,
CLI versions, the pre-freeze log prefix and origin/main parity before prospective
entry points. Preserve the original seed/output; never rerun to improve a number.
Update PROGRESS-HANDOFF.md after each step/item. Do not modify frozen instruments
after seeing held-out data; a defect then is an invalidation/disclosed limitation.

Record only supported CORROBORATED/FALSIFIED verdicts through manage_backlog;
UNASSESSED leaves the mission open. Preserve knowledge findings and limitations,
Chronicle history, evaluation lessons/interventions/repairs and a natural-observation
inbox candidate. Eight natural opportunities over 14 days are reported separately
and do not affect verdict; do not manufacture them. Selector adoption is a later
separate approval, not part of this experiment. Integrate final check-in before
local soul closure/export; commit export separately. Hugging Face remote sync
is not authorized. No SOUL PERSISTED claim without all required remote checks.
