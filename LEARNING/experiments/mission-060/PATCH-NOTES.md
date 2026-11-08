# Pre-freeze instrument repairs

## Freeze-preparation update

Thirty checks now pass. Composite worker/successor/reviewer cap accounting and
primary denominators replace the earlier worker-only accounting. Declarations
require full non-self coverage and actual integer values. Reviews require JSON
booleans; malformed evidence remains UNASSESSED. Unknown claim-A results cannot
corroborate, and a worker attempt without assessment cannot silently drop out.
Frozen-source/dependency/version/prefix checks and origin/main parity now gate
prospective execution; repeated held-out generation is refused. Review resolution
and all failure/recovery notes are preserved under verification/ and the progress
handoff. The unchanged calibration rubric passed 9/9 before this freeze. No real
held-out generator was executed by any test.

## Verification update — 2026-11-07 07:22 UTC

The live metering and worker write-containment requirements below have now been
verified. See PROGRESS-HANDOFF.md and verification/*.json for per-step failures,
repairs, source-field token audit, live backend responses and actual AGY terminal
canary. Nineteen regression tests pass. Validation used 14,140 additional tokens
over ten calls, including failures. No scientific calibration or held-out work ran.

New tracked launch wrappers suppress Codex/Claude child hooks for neutral-folder
integrity and extract their final assistant replies from attributed transcripts.
Flash tools run under the external macOS write-denial profile; native AGY sandbox
was tested and rejected for terminal incompatibility. Explicit write exceptions
are deliverables, per-worker runtime, AGY client state and PTY devices (/dev/null
also writable). Reads/network are not isolated. Preserve these disclosures and
include all new wrappers, profile builder and verification scripts in the freeze.
In-call token limits remain unimplemented; budget guards detect overruns between
calls. All changes remain local/uncommitted. The following sections preserve the
earlier repair state and its then-pending requirements.

Implemented locally on Richard's instruction to implement the patches and defects.
The study remains unfrozen. These engineering checks are not empirical evidence
for MISSION-060, and no study phase was executed by this repair task.

## Changes

- Flash workers, successors and formula scorer; Luna reviewers; Flash/Luna
  calibration. Declarers and idea generators are unchanged. Calibration requires
  different-family raters and actual JSON boolean judgments.
- Worker rotation supports a one-client list and rejects an empty list.
- Heartbeat renews the `codex` lease and propagates renewal failures.
- Quota probes fail closed on missing backend readings. The existing ANY-backend
  25% rule and scientific numerical thresholds are unchanged. Missing Claude
  study-share baseline or current measurement also stops Claude calls.
- Transcript attribution requires one new transcript containing a unique tag
  included in that call's input. Existing, absent or ambiguous matches are
  unknown. The tag is an operational input change to disclose before freeze.
- Unknown or invalid usage is never imputed as zero. The next dispatch stops
  on unknown usage, a reached total ceiling or a worker overrun. Status reports
  known subtotal separately. Analysis cannot issue a scientific verdict when
  any logged study call has unknown usage. Duplicate call IDs cannot be reused.
- A changed repository status after worker execution stops further processing.

## Verification

From this directory: `python3 -m unittest test_instruments -v`.

All 15 tests passed using temporary development fixtures and mocked calls.
Python compilation, JSON parsing and `git diff --check` also passed. The first
13-test run passed but emitted unclosed-file warnings; touched paths were
corrected, and the final 15-test run emitted no such warnings.

## Required before freeze or held-out execution

- Validate each backend's metering against the first actual development call.
  Fixture checks establish parser behavior, not live transcript compatibility.
- Budget guards stop before/after calls; they do not interrupt generation at a
  precise token cap. Do not describe these as enforced in-call token ceilings.
- A git status comparison does not detect every content change to an already
  dirty tracked file. Snapshot copying does not itself prove OS-level read-only
  containment. Worker containment remains unverified and must be resolved before
  execution, as required by the handoff.
- Preserve the original calibration-before-freeze order, separate freeze
  integration, quota pause rule, and no-Hugging-Face-sync boundary. No freeze
  hashes, calibration results or mission verdict were created by this task.
- Include `test_instruments.py` and these implementation disclosures in the
  freeze inventory alongside every original handoff-required file.

Delivery is local, uncommitted and unmerged in `feat/codex-mission-060`.
The mission lease remains held by `codex`; this is not mission completion.
Repair-task token cost was not assessed; prior preparation dispatch usage is
unknown. No measured savings or ROI is claimed.
