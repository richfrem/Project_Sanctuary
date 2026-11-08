# MISSION-060 progress and AGY handoff

## Step 4 freeze candidate prepared — 2026-11-08

protocol.md and freeze.json now exist. Freeze verification PASS: 37 frozen
repository files and 31 runtime dependency files, CLI/Python versions and immutable
pre-freeze usage prefix. All 30 deterministic checks PASS; all 62 manifests valid;
generated index/backlog compiled with the supported tool. Credential-pattern scan
matched no files. Integration-required freeze verification correctly REFUSED before
public source integration (expected negative check; no held-out evaluation ran).

Pre-freeze fixes: item caps/B1 denominators include all three stages; declaration
coverage/types validated; malformed review evidence is UNASSESSED; unknown A
cannot corroborate; attempted worker records cannot disappear from analysis.
Prospective entry points require integrated freeze and held-out generation refuses
an existing result. Numeric thresholds and calibrated rubric unchanged; worker
instruction clarifies composite cap. See verification/freeze-review-resolution.md.

The old 36%-remaining baseline was archived; current pre-freeze Claude session
baseline is 100% from the saved live snapshot, with concurrency/reset limitations
disclosed. Scope narrowed to the experiment directory and mission manifest.
One freeze-create invocation used the known incomplete worktree shell PATH and
failed before writing freeze.json; the subsequent verification found no file.
Rerunning from the documented main shell succeeded. Preserve this as an operator
mistake, not a tooling defect. Always invoke study scripts by absolute path FROM
the main shell/working CLI PATH. No scientific calibration or held-out run repeated.

Next: commit this separate freeze package, integrate via PR, fetch/verify identical
files on origin/main, run required regeneration, then refresh quota before step 5.
The freeze is NOT publicly effective until that verification passes. No step-5
artifact has been generated; do not start it before integration.

## Current status — 2026-11-08 05:37 UTC

User checklist: **1 complete, 2 complete, 3 complete; 4–6 not started.**
Calibration first run passed 9/9 paired judgments (1.0 >= 0.80), no revision.
Companion parser fix merged in PR #757 at 6053a2ec and globally installed; three
harness copies verified, 33 installed tests pass. Generic control plane disabled
and consistent under explicit approval. Mission work remains uncommitted in
feat/codex-mission-060; keep this worktree. Next: step 4 protocol/freeze preparation
and separate integration before ANY held-out execution. Quota needs refreshing
at that next phase. Current measured study-call total 36,561 additional tokens;
calibration portion 22,421. Whole-task orchestration/delegation cost is unknown.

Below are the completed-step updates and original assessment history; superseded
pending-decision or not-started claims are historical. Current stop point/table
near the end are updated to this status. No mission verdict or soul closure yet.

Companion cleanup completed after verification: feature worktree was clean and
`git diff c93f4b7b 6053a2ec` showed identical retained trees. Removed the companion
feature worktree and local/remote feat/usage-parser-resetless branch. Git's local
delete warned that original c93 was not an ancestor of main (expected after
squash); retained content was verified against public merge 6053a2ec first. The
companion is on clean merged main; earlier companion-worktree paths are historical.
The Sanctuary mission worktree and its uncommitted artifacts remain intact, with
Codex lease active to 06:28 UTC. No cleanup of the mission worktree occurred.

05:37 UTC final calibration accounting: source-field usage audit PASS for all
16 logged validation/calibration calls, zero unknown entries. Calibration used
22,421 additional tokens across six calls (Flash 21,980; Luna 441); scaffold
114,558 and output 8,439 reported separately. Cumulative measured study calls:
36,561 additional tokens. Broader orchestration/Haiku costs remain separate and
unknown; no total-cost or savings claim. git diff --check PASS. Next is step 4:
write/finalize protocol and freeze inventory, narrow manifest impact, and integrate
the separate freeze before any held-out execution. Mission is not complete.

## Step 3 COMPLETE — 2026-11-08 approximately 05:36 UTC

First live Flash/Luna development calibration completed successfully:
calibration-results.json records agreement=1.0, agree=9, total=9, exceeding the
unchanged 0.80 bound. All D1/D2/D3 judgments on cases 056/057/058 were true for
both raters. No prompt revision, repeat or held-out tuning was used. Six calls
are retained under calibration in usage-log.jsonl and raw/cal-*.out/result.json.
This is agreement on three positive development cases, not evidence of evaluator
specificity, objective ground truth or the mission hypothesis. Source-usage audit
is next; no protocol freeze, held-out simulation, declarations or items have run.
Step 2 remains COMPLETE (public merged fix installed/verified, consistent disabled
generic control plane, valid fresh quota). Do not repeat the old pending decisions.

## Step 2 COMPLETE; step 3 starting — 2026-11-08 05:33 UTC

PR #757 merged at public 6053a2ec; clean companion main fast-forwarded. Approved
`apm update --global --yes` completed (seven dependencies refreshed; log saved at
/private/tmp/mission060-apm-update-20261108.log). Full evaluating-agent-outcomes
skill comparisons against merged source show no differences for ~/.claude/skills,
~/.agents/skills and ~/.gemini/antigravity-cli/skills, excluding Python caches.
All 33 INSTALLED tests independently pass. The live reset-less Claude display now
parses as session_pct_used=0, session_resets=null, status=ok. Global dependency
listing saved at /private/tmp/mission060-apm-deps-20261108.txt; APM manifest backups
remain at /private/tmp/mission060-apm-backup-20261108-0528/.

Generic control plane remains disabled/consistent under Richard's approval.
Fresh native Codex account read at 05:33:33 UTC: 86% five-hour, 31% weekly
remaining, ordinary usage allowed; saved in verification/codex-calibration-quota-20261108.json.
Claude tightest 35%, AGY 49.857%: all above 25%. The older rollout collector
reports Codex 34%; this is superseded by the conservative live 31% reading for
this phase. Driver status succeeds: 10 calls / 14,140 known additional tokens,
no unknown validation calls. Broader orchestration/delegation cost remains unknown.

Beginning development calibration now: Flash/Luna, cases 056/057/058, nine paired
boolean judgments. Pre-run prediction and rules are in CALIBRATION-PREFLIGHT.md.
No freeze or held-out execution authorized by this step. Update immediately after
calibration with agreement, usage and any failures; do not claim success early.

05:29 UTC deployment step complete: PR #757 passed both GitHub checks and was
squash-merged at 6053a2ec9fe3474bb9c5d2a72261aeb804307c58. Clean companion main
fast-forwarded to that public commit. APM manifests backed up under
/private/tmp/mission060-apm-backup-20261108-0528/. Running a post-merge global
update preview before applying the approved install; no installed parity claimed
yet. The companion fix is publicly merged; Sanctuary mission changes are not.

## Approved deployment in progress — 2026-11-08 05:28 UTC

Richard explicitly approved publishing/merging the reviewed companion parser fix
and global installation. Pushed c93f4b7b on feat/usage-parser-resetless using the
Protocol 133 companion workflow (pre-push skip per that workflow; prior 33 tests
and audit passed). Opened https://github.com/richfrem/agent-plugins-skills/pull/757.
CI is running; no merge or install yet. Main remains 7a4c29eb. APM global update
dry run completed without changes. Next: wait for checks, merge, fast-forward
clean companion main, back up APM manifests, update globally, verify installed
parser and quota gate, then run development calibration. No held-out work.

## Resume update — 2026-11-08 approximately 05:20 UTC

Richard requested completing step 2 and beginning calibration. Control plane
reverified disabled/consistent; Codex lease renewed to 06:17:53 UTC. Installed
Claude parser again returns unparsed on reset-less usage, so the automated gate
still cannot pass. Companion fix c93f4b7b remains local; remote main verified
7a4c29eb and no PR exists. Asked one direct authorization question for push,
merge and global installation of the reviewed 33-test fix. Await response; no
publication or installation performed. APM update --global --dry-run completed.

Codex account/rateLimits/read refreshed with no model call: 97% five-hour and
33% weekly remaining, ordinary usage allowed. Sanitized response retained in
verification/codex-current-quota-20261108.json; do not use stale rollout readings.
Calibration preflight prepared in CALIBRATION-PREFLIGHT.md with the command,
development-only cases, rater pair, gates and prospective prediction. Actual
calibration has NOT started; no new study calls or freeze. Next: deploy shared
fix within explicit authorization, verify installed behavior, then run calibration
and update this file with actual agreement and usage. This does not reopen the
already-resolved control-plane decision.

## Control-plane decision applied — 2026-11-07 18:03 UTC

Richard explicitly approved: "yes disable for this repo". Applied the supported
`control_plane_mode.py disable --target /Users/richardfremmerlid/Projects/Project_Sanctuary --yes --actor Richard`.
Result: DISABLED and consistent. Independent status checks from both main and the
mission worktree returned mode=disabled, consistent=true, empty problems/warnings,
and both generic guards unwired. Evidence:
`verification/control-plane-disabled-20261107.json`. Backup:
`context/control-plane-backup/20261107-180351` in the main checkout.

The control-plane decision is resolved; older pending-decision notes below are
historical. This is a per-clone machine-local setting shared by this clone's
worktrees, not a global skill uninstall or a change to Sanctuary's Human Gate,
flight-board coordination, explicit push/merge authorization or signing trust.
No database, retrofit, key, calibration or held-out operation was performed.
No model call was needed for this configuration step; orchestration cost unknown.

Step 2 follow-through: quota was above the floor on Haiku's fresh reads, but must
be checked again before phase 3. The defensive parser fix remains committed only
in the companion feature worktree and is not installed. Its publication/merge/
global update is a separate authorization boundary, not permission granted by
this control-plane decision. Do not continue reporting a pending mode choice.

## Haiku delegation outcome — 2026-11-07, approximately 17:51 UTC

Claude CLI Haiku 5.5 completed the bounded step-2 task in 336.776 seconds (router
exit 0; model tokens not reported). See HAIKU-STEP2-RESULT.md, the saved work
contract and verification/haiku-step2-dispatch-result.json. Step 2 is still PARTIAL,
not complete: live quota evidence refreshed, parser source repaired locally,
generic control-plane decision and source deployment still outstanding.

Fresh native account readings: Claude tightest 35%, Codex 37%, AGY approximately
50%, all above 25%. Codex evidence came from app-server account/rateLimits/read,
not the stale rollout collector. Source repair is committed as c93f4b7b on
companion branch feat/usage-parser-resetless at
/Users/richardfremmerlid/Projects/agent-plugins-skills/.worktrees/usage-parser-resetless.
It retains displayed percentages when reset suffixes are absent. Root reviewed
the source diff, verified the worktree is clean and independently reran all 33
companion tests: PASS. Exported patch is in verification/haiku-usage-parser-fix.patch.

No push, PR, merge, global install, mode change, key enrollment or study phase.
Installed parser still has the reset-less defect; live CLI output presently has
a reset suffix and parses. Quota must be rechecked before calibration. Pending
decisions: control-plane disable-unused vs compatible setup, and authorization
for companion publication/integration/install. Do not read delegation as approval
for either. Codex lease was renewed by root to 18:45:41 UTC (delegate's older
17:54 lease warning is stale). Sanctuary worktree changes remain uncommitted.

Accounting clarification: the delegate was itself one real Haiku model session;
its "no provider model calls" wording refers to its quota utility reads and no
nested model dispatch, not zero cost for the delegate. Its token total is unknown.
Earlier 14,140 measured validation tokens exclude this administrative repair
dispatch and must not be presented as full mission/orchestration cost.

## Haiku delegate update — 2026-11-07, approximately 17:44–17:55 UTC (PARTIAL)

Full result: `HAIKU-STEP2-RESULT.md`. Quota evidence: `verification/haiku-quota-20261107.json`.

- **Quota (fresh, no model calls):** Claude session 100% and week 35% remaining
  (live output has a reset suffix; the reset-less form is still unparsed by the
  installed parser). Codex app-server read at 17:50:08 UTC: 5h 88%, week 37% remaining
  (supersedes the stale 07:14 rollout). AGY smallest bucket 50% remaining. All above the 25% floor.
- **Parser defect repaired locally, not installed:** companion repo
  `feat/usage-parser-resetless` at `c93f4b7b` (worktree
  `agent-plugins-skills/.worktrees/usage-parser-resetless`, base `main` 7a4c29eb). Tests
  32 → 33 OK, audit PASS. Patch: `verification/haiku-usage-parser-fix.patch`. Not pushed,
  merged or installed. Installed copies still match the old parser.
- **Control plane:** status still ENABLED (default, no mode file) and INCONSISTENT. Guards not
  wired, no database. Applicability is an open human decision; no mode change. Not passed.
- **Failures recorded:** an unread-file Edit refusal, a no-op test edit, and a five-cell
  evolution-log row. All corrected. Pre-commit guard warned "ungated" in the companion clone.
- **Usage:** delegate model tokens unknown; no provider model calls made.
- **Lease:** codex owner, not touched. Gates file recorded renewal to 17:54:31 UTC, which is
  near. Root agent must heartbeat before further work.
- **Next:** human decides disable-unused vs keep-enabled for the generic plane; human decides
  companion push/PR/merge and `apm update --global`; renew Claude sign-in before 2026-11-10.
  Recheck quota before calibration. Do not run calibration, freeze or held-out work yet.

## Step 2 current update — 2026-11-07, approximately 16:54 UTC

Step 2 remains PARTIAL. See `verification/gates-20261107.md` and the saved quota
snapshot. Claude raw usage shows 100% session/35% weekly remaining, but the shared
parser mishandles the reset-less 0%-used display. Codex collector data is stale;
do not claim its old headroom is a fresh measurement. AGY is above the floor.
Generic plane was never initialized (installation probe FRESH); full retrofit
conflicts with canonical top-level domains and human signing is not enrolled.
Supported enable/disable dry runs completed; user decision on disabling the unused
generic plane vs preparing compatible setup is pending. No mode changed. Lease
renewed under codex to 17:54:31 UTC. No new model dispatch, calibration or freeze.

## Current summary — 2026-11-07 07:22 UTC

**Requested step COMPLETE:** live token metering verified on Flash, Luna and
Sonnet; actual Flash terminal-tool write containment verified. Ten development
calls used 14,140 additional tokens (including failed attempts/repairs); no
unknown usage among those calls. `verification/usage-source-audit.json` recomputes
the logged values from numeric transcript fields and verifies attribution tags.
Final 19 regression tests, Python compilation and git diff checks PASS.

Resume in the worktree/branch below. All changes are still uncommitted/unmerged.
Codex lease renewed to 08:22:17 UTC. Next scientific step is development rubric
calibration, subject to fresh quota checks and the unresolved generic control-
plane workflow question. Do not rerun successful metering/canary probes merely
to fill time. No calibration, held-out evaluation, freeze or verdict has run.

Containment scope: the external macOS profile denies filesystem writes except
deliverables, per-worker runtime, AGY client-state files, /dev/null and specific
PTY devices. Reads and network remain available; client state is writable. AGY
native `--sandbox` was not compatible with its terminal path here; the supported
auto mode runs inside the verified external profile, with permission-bypass
flags stripped. This is not a general hostile-agent or data-exfiltration boundary.
In-call token-cap enforcement is still not implemented; existing guards stop
subsequent calls after a measured overrun. Earlier prep-dispatch and orchestrator
token cost remain unknown; 14,140 is only the measured validation-call total.

## Verification history (newest first; earlier pending states are historical)

07:22 UTC actual AGY worker containment PASSED: `worker-tools-04` executed the
terminal canary under the external OS profile. Deliverable write allowed,
snapshot and outside fixture writes denied, both protected files unchanged,
probe artifact preserved with SHA-256. Additional usage 3,053 tokens. See
`verification/worker-tools-04.json` and `worker-tools-04-probe.json`. Current
profile native tests: nine PASS in `containment-integrated-04.json`. Earlier
failed canaries remain evidence and are included in usage totals. Metering and
write containment are now verified for these development paths; final source-
field usage audit and regression checks follow before reporting this step done.

07:21 UTC `worker-tools-03` FAILED at AGY's own terminal sandbox_apply with exit
71 (Operation not permitted), after initialization requirements were met. Files
unchanged; 3,446 additional tokens charged. A direct nested sandbox-exec check
was insufficient to establish compatibility with AGY's terminal implementation.
The wrapper now keeps the verified external OS write profile as its enforcement
layer and uses supported AGY `--mode auto`, without `--sandbox` or permission-
bypass flags. Removed the now-unneeded `~/.gemini/sandbox` write exception.
This is a disclosed pre-freeze launch change, not a claim that AGY native sandbox
passed. Repeat native write-denial checks and one new-ID end-to-end canary.

07:20 UTC native nested-sandbox check exited 0. Added the exact AGY native
profile-state directory `~/.gemini/sandbox` as a write exception; did not allow
all of `.gemini` or home. `containment-integrated-03.json` PASSED all nine native
checks under the revised profile. Both backend state directories and PTY devices
are explicit exceptions. New canary `worker-tools-03` will exercise the actual
native AGY terminal path with repository/snapshot protections retained.

07:19 UTC `worker-tools-02` FAILED after the PTY repair: native AGY sandbox
initialization could not create `~/.gemini/sandbox`. Protected fixtures unchanged,
no successful write artifacts; 2,997 additional tokens charged. This is a distinct
runtime-state exception requirement, not successful containment readiness.
Investigate nested-sandbox compatibility deterministically before more calls;
do not add broad home-directory write access or remove the external OS boundary.

07:18 UTC PTY repair validated: installed Apple sandbox profiles document
`pseudo-tty`, `/dev/ptmx` and `/dev/ttys[0-9]+` device allowances. Added only those
device operations. `verification/containment-integrated-02.json` PASSED nine
checks, including live PTY creation and all previous file-write denials. Next:
new-ID `worker-tools-02` canary through the actual AGY terminal tool chain.

07:18 UTC tool-enabled canary `verification/worker-tools-01.json` FAILED:
Flash truthfully replied BLOCKED, no deliverable was created, and both protected
fixtures stayed unchanged. Its transcript contains two terminal-tool errors:
"failed to create PTY: operation not permitted." Measured additional usage:
3,486 tokens, retained in usage-log.jsonl. Startup-only containment PASS is not
worker readiness. Investigate a minimal PTY exception; retain repository/snapshot
write denial and repeat native checks before any new-ID tool canary. No study
item or calibration ran.

07:17 UTC contained Flash startup `meter-flash-contained-01` PASSED under the
actual worker executable: exact response, fresh metered transcript, 583 additional
tokens; profile text/hash saved in its verification report. This is an isolated
startup probe, not evidence that model tool writes work. Next is one bounded
tool-enabled development canary against disposable files, then source-usage audit.

07:16 UTC Sonnet repaired-path probe `verification/meter-sonnet-02.json` PASSED:
1,897 scaffold tokens, 8 additional/output tokens, exact reply, fresh attributed
transcript and hash, neutral directory still empty. Live measurement has now
passed for all three study backends. Next: contained Flash startup smoke check,
independent source-field arithmetic audit, final regression checks and summary.

07:16 UTC Sonnet first probe: live metering succeeded (24,100 scaffold and 8
additional/output tokens), but the overall check BLOCKED. A CLI sign-in warning
prefixed the exact model response, and Claude Stop hooks wrote neutral/context.
Preserve that hook artifact outside neutral. The repair uses native Claude
`--safe-mode` (retains existing auth, disables custom hooks/context), tools disabled
and strict MCP config; canonical response is read from the attributed transcript.
Do NOT use `--bare`: installed help says it bypasses OAuth/keychain auth, which
would change the authorized subscription path. Probe IDs/failures remain intact.

07:14 UTC completed steps: Luna repaired-path probe `meter-luna-03` PASSED
(21,622 scaffold, 8 additional/output tokens, exact response, unique hashed
transcript). Native containment using the actual worker profile builder PASSED
all eight checks in `verification/containment-integrated-01.json`, including
repository write denial and runtime write allowance. The worker wrapper now
exists and is wired into tool-enabled Flash calls, strips permission bypass and
requests the native AGY sandbox. Its live startup still requires a smoke check.
Write exceptions are deliverables, per-worker runtime and AGY client state;
reads/network are not isolated. The initial native check had no client-state
exception; the integrated profile's separate check preserves that distinction.

Initial handoff created: 2026-11-07 07:06 UTC by Codex.

Completed live Flash check: `verification/meter-flash-03.json` is PASS: exact
METERING_OK response, unique new transcript and hash, 2 requests, 12,891 scaffold
tokens, 543 additional tokens, 26,149 accumulated input and 176 output tokens.
The real development call is charged in `usage-log.jsonl` under
`instrument-validation`, not counted as a discovery or calibration result.
Sonnet and Luna live checks and worker-launch integration remain pending.
Quota recovered after reset (initial verification snapshot at 07:08:38 UTC);
the old quota blocker must not be treated as a current reading.

Luna live check `verification/meter-luna-01.json`: token attribution PASSED
(one fresh tagged transcript, 21,715 scaffold tokens, 8 output/additional tokens),
but response-contract check FAILED. Router output includes echoed input, hooks
and CLI diagnostics rather than just the reply. The transcript's final assistant
message is exactly METERING_OK. `common.py` now extracts Codex final assistant
text from the attributed transcript, retaining raw stdout as evidence. This also
prevents accidentally parsing echoed declaration-input JSON as model output.
A disclosed new-ID Luna probe will verify the repaired full path; the initial
failure remains preserved. No held-out work ran.

`meter-luna-02` stopped before dispatch: the first Codex call's Stop hooks had
written `neutral/context/events.jsonl`, violating empty-neutral-directory checks.
The file contains two session-summary hook events, not a study artifact. Preserve
it outside the neutral directory. Native `codex features list` reports `hooks`
stable and enabled; use supported `--disable hooks` in the tracked, per-dispatch
`codex_medium.sh`. This changes only isolated study child calls, not global or
orchestrator configuration, sandbox or approval policy. Test the repaired path
under a new call ID and verify the neutral directory stays empty afterward.

Latest verification update (2026-11-07 07:10 UTC): native OS containment probe
completed with six checks passing; see `verification/containment-native-01.json`.
Deliverable writes succeeded; snapshot, outside, symlink-escape and descendant
writes were refused; protected fixtures stayed unchanged. This proves the tested
native profile, not integration into actual AGY worker launches. Live probe
`meter-flash-01` stopped BEFORE dispatch because Claude quota was unavailable
when queried from the worktree. The same quota command from the main checkout
returned valid readings (Claude 26%, AGY 49.9%, Codex 42% tightest remaining).
No model tokens were spent by that blocked attempt. Resolve quota probe working
directory explicitly; do not bypass the missing-reading gate. Evidence:
`verification/meter-flash-01.json`. Live study instrumentation validation is now
authorized by Richard; calibration and held-out phases remain unstarted.

Update this file immediately after each completed step, before starting the next.
For each update record the action, evidence, failures/recovery, remaining limits,
next action, and current workspace/lease state. Do not mark a step complete merely
because its draft exists. This is the live execution handoff; scientific findings
and episodic Chronicle records still belong in their required separate records.

## Resume location

- Repository: `/Users/richardfremmerlid/Projects/Project_Sanctuary`
- Worktree: `/Users/richardfremmerlid/Projects/Project_Sanctuary/.worktrees/codex-mission-060`
- Branch: `feat/codex-mission-060`
- Current committed HEAD: `f469f62ba0d198979dfbd0224da4a98e27077740`
- Study directory relative to worktree: `LEARNING/experiments/mission-060/`
- Changes are local, uncommitted and unmerged. Preserve tracked edits AND untracked
  files. Do not reset, remove the worktree, or start another checkout for this work.
- Flight board verified at this update: active MISSION-060 lease owned by `codex`.
  AGY must verify current ownership and use the supported coordination workflow
  before taking over. Do not assume a historical lease is still valid or renew a
  different owner's lease. No AGY takeover has occurred.
- Main had an unrelated modified `.agent/learning/guardian_boot_digest.md` at
  bootstrap; leave unrelated main-checkout edits untouched.

## Read in this order

1. `exploration-policy-revision.md` and `HANDOFF-codex.md` (original approved
   design, sequence and operational disclosures; section 9 records local repairs).
2. This file and `PATCH-NOTES.md` (actual progress and limitations).
3. The source changes and `test_instruments.py`.
4. `flash-*-draft.md` and `flash-preparation-plan.md` only as annotated proposals.
   They contain errors; they are not adopted protocol or verified findings.

## Completed steps and evidence

| Step | State | Evidence and result |
| --- | --- | --- |
| Bootstrap and continuity | Complete | Constitution and Layer 1 files read; existing hook satisfied briefing; git status checked; mission handoff and revision read. No second wakeup run. |
| Isolated mission checkout | Complete | Guardian checkout created this branch/worktree and marked manifest IN_PROGRESS. Board confirms owner `codex`; heartbeat renewed during work. |
| Initial quota assessment | Complete; stale for execution | Headroom and snapshots taken. Last recorded snapshot at 06:33:59 UTC showed Codex 18% remaining, Claude 26%, Gemini five-hour 99.769%. These are historical readings, not permission to dispatch now. |
| AGY preparation dispatch 1 | Complete | `flash-preparation-plan.md` and `flash-preparation-result.json`; AGY `gemini-3.8-flash-medium`, exit 0, 74.761 seconds. Confirmed heartbeat owner and single-worker modulo defects. Rejected unauthorized quota-rule weakening and compiling JSON as Python. |
| AGY preparation dispatch 2 | Complete | `flash-prefreeze-package-draft.md` and `flash-prefreeze-package-result.json`; proposed patch, protocol draft and verification checklist returned. Annotations identify wrong source line numbers, SPO wording, unsupported containment claims and incomplete freeze inventory. Not applied verbatim. |
| Instrument repairs | Complete locally | Flash workers/successors/scorer, Luna reviewers, Flash/Luna calibration; one-worker rotation; checked Codex heartbeat; fail-closed quota and usage; uniquely tagged fresh-transcript attribution; duplicate call-ID refusal; stop on repository-status change. See PATCH-NOTES.md. |
| Deterministic repair verification | Complete | Final `python3 -m unittest test_instruments -v`: 19 passed, no warnings. Python compilation, thresholds JSON parsing and git diff whitespace checks passed. Regression tests used temporary development fixtures and mocked model calls; separate live probes are recorded below. |
| Live metering and worker containment | Complete for tested scope | Passing meter-flash-03, meter-luna-03 and meter-sonnet-02; nine current native profile checks; actual AGY terminal canary worker-tools-04 PASS. Source-token audit PASS for all ten development calls, 14,140 additional tokens. See verification/*.json and current summary. |
| Live progress handoff | Complete | This file created on Richard's instruction; original handoff points here. Update after every subsequent completed step. |

The initial 13-test run passed with unclosed-file warnings. Touched file handling
was corrected; the final 15-test run passed without those warnings. This records
the recovery rather than erasing it. No actual model metering was validated by
these tests.

## Current stop point and next action

**Development calibration is complete: 9/9 agreement, no prompt revision. No
freeze, held-out scenarios, declarations, item execution, analysis or verdict
exists.** The mission remains IN_PROGRESS.

Next: finalize protocol/freeze inventory, narrow manifest impact, then integrate
the freeze separately before held-out execution. Refresh quota before that phase.
Do not weaken the ANY-backend 25% stop rule. Do not repeat successful metering,
containment or calibration without a new justified failure/change.

Verification recovery at 07:12 UTC: `meter-flash-02` also stopped before dispatch.
Diagnosis refined: Claude probe raised FileNotFoundError because shells launched
in the worktree study directory lack `~/.local/bin` on PATH; setting subprocess
cwd alone did not fix inherited PATH. The main-checkout shell resolves the real
Claude binary. Run the probe script by absolute path FROM MAIN (without changing
the main checkout's files) to inherit the working CLI PATH. `meter-flash-03` is
the first attempted live dispatch after the two zero-call gate refusals. Do not
reinterpret the prior refusals as model failures or silently discard them.

### Gates and unresolved technical limits

- Quota previously tripped the approved study stop rule. Richard authorized Flash
  workers and patch implementation, but did not explicitly change the quota rule.
- Generic control plane is explicitly disabled and verified consistent under
  Richard's approval; no unresolved mode decision or signing gate for this phase.
  The shared parser fix is merged/installed/verified. Refresh quota for new phases.
- Metering passed on all used backends with unique transcripts and a source-field
  audit. Missing/ambiguous usage still stops progress. Earlier operational
  preparation-dispatch usage remains unknown and separate.
- Worker write containment passed at OS and actual AGY-terminal levels; use the
  tracked contained worker wrapper, not snapshot copying or git status alone.
  Respect the explicit write exceptions and read/network limits in the summary.
- Budget checks stop before/after calls, not during generation. They detect an
  overrun and block subsequent calls; they do not enforce a precise in-call cap.
- Driver heartbeat is currently fixed to owner `codex`. A formal AGY takeover
  must align driver ownership before execution; do not silently spoof the owner.

## Remaining sequence — update this table after each step

| Order | Step | State | Completion evidence to add |
| --- | --- | --- | --- |
| 1 | Fresh quota/lease checks; resolve workflow gates | Complete | Mode disabled/consistent; parser merged/installed; all quota above floor before calibration. Recheck before next phase. |
| 2 | Validate live metering and resolve containment/cap limitations | Complete for tested scope | See current summary and verification reports. Exact in-call cap remains a disclosed limit; reads/network and client state are not isolated. |
| 3 | Calibrate on 056/057/058 with Flash/Luna | Complete | calibration-results.json: 9/9 agreement, 1.0 >= 0.80, no revision. Six calls, 22,421 additional tokens; source audit PASS. |
| 4 | Final protocol, narrow manifest impact, freeze inventory | Pending | All 12 original operational disclosures plus implementation changes; hashes of every required source/config/base file and protocol. Include new tests and disclosures. |
| 5 | Commit and integrate freeze separately | Pending | Approved integration and actual commit/PR references, before any held-out call. No direct main commit or protected-main push. |
| 6 | Claim A held-out scenarios once | Pending | Frozen seed/output hashes and result; never rerun to improve scores. |
| 7 | Prospective snapshot, ideas, score, declare, choose | Pending | Usage log, blinded declarations and attestation, selection; preserve exact order. |
| 8 | ITEM-01 through ITEM-10 sequentially | Pending | Per-item worker/successor/reviewer outputs, assessments, usage, limits and interventions. Update this file after each item. |
| 9 | Analyze and record supported verdict | Pending | results.json and per-claim evidence; UNASSESSED leaves mission open, not FALSIFIED by default. |
| 10 | Knowledge, Chronicle, lessons and natural-observation inbox candidate | Pending | Paths/IDs; observation is separate, not manufactured and not verdict-gating. |
| 11 | Final check-in, source integration, local soul closure/export | Pending | Lease/handoff, integration refs, schema/coverage/export checks and scoped export PR. HF remote sync is not authorized. |

## Cost, delivery and preservation

Preparation dispatch usage was not reported by the router and remains unknown.
Repair/orchestration total token cost was not assessed. No savings or ROI claim.
Ten instrument-validation calls total 14,140 additional tokens; six calibration
calls total 22,421. All 16 log records source-audited, 36,561 additional tokens,
no unknown entries. Calibration agreement is a development reliability measure,
not a mission verdict. No freeze, held-out selection or study items have run.

The companion parser fix is publicly merged and globally installed. Sanctuary
mission artifacts remain local/uncommitted/unmerged. No soul export or remote
Hugging Face sync was performed or authorized. Do not claim mission completion
or SOUL PERSISTED. Companion feature worktree/branches cleaned after verifying
retained content; Sanctuary mission worktree/lease retained.

## Per-step update format

Append a dated entry after the action and update the tables/status above:

```text
UTC time — Step/item — COMPLETE / PARTIAL / BLOCKED
Action and changed artifacts:
Evidence and checks:
Failures/recovery and unresolved limits:
Usage/cost (measured or unknown):
Worktree/branch/HEAD and lease owner/state:
Exact next action and what not to repeat:
```
