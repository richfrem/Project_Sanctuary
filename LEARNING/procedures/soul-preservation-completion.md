# Mission completion and soul preservation integration

## Repair scope and prediction (2026-11-06)

Continue the already merged coverage/backfill repair without new missions or research.
Prediction: an atomic obligation in the final manifest, serialized local completion,
and startup retries can close the manual-capture gap without rebuilding extraction
or transport. Deterministic fixtures test this prediction; verdicts remain independent.

## Verified behavior

`record-result` saves the final disposition and obligation together for completed evaluations (`CORROBORATED`, `FALSIFIED`, or `ABANDONED`). Interrupted or unassessed closeouts preserve `"verdict": null` without manufacturing a verdict. After sources are
integrated, run `merge_prep.py regenerate` in a checkout containing their published
commits (normally updated main). It runs the existing incremental export and validation.
A feature checkout whose artifact commits were squash-merged may need a fresh branch
from `origin/main` before references become eligible. `soul_coverage.py closure` resumes;
Guardian boot retries eligible pending work. No automatic git/remote operations.

70 bounded tests passed across coverage, preservation workflow, backlog, merge preparation,
Guardian persistence and hooks. Fixtures verified findings, limitations, final manifest,
key-file hashes/references, unpublished-to-published retries, export failures, process
interruption, simultaneous completions and exports, stable repeated closure, receipt
exclusion, both remote hashes, and preservation/scientific separation. No experiments rerun.

The existing extractor and synchronizer were reused. Dataset and coverage replacements
are atomic; a shared OS lock serializes supported local writers. Local status lives in
ignored coordination state and is reconstructed from tracked obligations and export hashes.
Export-only/receipt commits do not stale learning. Arbitrary external merges are detected
at startup and by staleness checks, rather than intercepted.

## Protocol & Rule Alignment (2026-11-08)

Aligned with `.agent/rules/mission-execution.md` (ratified 2026-11-08) and `coverage.md`:
1. **Separation of Scientific Outcome and Delivery**: The empirical verdict (`record-result` for completed evaluations, or `verdict: null` with `UNASSESSED` closeout note) is distinct from engineering delivery (PR merge) and preservation state (`soul_coverage.py check`).
2. **Check-In Before Capture Ordering**: `manage_coordination.py checkin` appends to the feature branch's tracked `.agent/learning/events.jsonl`. Commit the final check-in on the feature branch and merge it to `origin/main` before running `merge_prep.py regenerate` / `soul_coverage.py closure`, so final learning events are included in the capture.
3. **Non-Recursive Export Publication**: Publish the resulting export (`huggingface/`) and sync receipt (`.agent/learning/soul_sync_receipts.jsonl`) in a scoped follow-up PR without appending another check-in that would re-dirty watched sources.
4. **Remote Verification**: Verify both remote dataset and coverage manifest SHA-256 hashes via `sync_soul_dataset.py` before claiming `SOUL PERSISTED`. Local capture alone never grants remote authority; startup observes read-only while capture runs at closure.

## Bootstrap hook friction

Configured Codex 0.167.0 user hooks: two agent-agentic-os SessionStart invocations and
two exploration SessionStart invocations; project Guardian SessionStart is also configured.
Before repair, each exploration command exited 0, stdout was a 468-byte JSON object with
`additionalContext` AND `hookSpecificOutput`, stderr empty. Each OS invocation exited 0,
stdout `{"continue": true}`, stderr empty. Guardian returned nested hook-specific context,
exit 0, stderr empty. The repeated two errors came from the exploration fallback adapter.

Companion PR 756 (main 7a4c29eb) fixes that adapter, preserving explicit Copilot/Cursor
formats. No hooks disabled. APM global update refreshed the source-managed installation.
After repair both exploration invocations exit 0 with 263-byte nested-only JSON and empty
stderr; all four global commands have valid supported keys. Installed hook byte-matches
companion source. A fresh Codex 0.167.0 session exited 0, returned HOOK_SMOKE_OK, and reported
zero invalid SessionStart JSON errors. Contract: https://learn.chatgpt.com/docs/hooks.

The generic control-plane probe found no initialized database/guards in Sanctuary;
this repair uses its existing authorized flight-board/worktree workflow and does not
bootstrap a new control-plane architecture. The unrelated evaluation-reminder branch
and main checkout boot-digest edit were preserved.

## Investment evaluation

Prediction corroborated by bounded fixtures: reuse plus a durable workflow closed the
capture gap. Additional discovered defects were the mixed hook output and receipt checker
not binding its coverage-manifest hash; both fixed. Exact host-session tokens are unknown;
the bounded fresh-session smoke measured 23,754 input and 32 output tokens (23,786 total),
recorded through track_budget.py. Flight-board snapshots provide account usage where available. No token estimate is represented as a measurement. Contribution
score and usage events are recorded with the existing evaluation tools before final export.

## Installed post-integration verification

Sanctuary source PR 378 merged at 4c3c6c3bc. Project APM refreshed; all 29 Python
scripts across preservation, sovereign memory and Guardian match source in both
`.agents/skills` and `.claude/skills` (APM rewrites two documentation links).
The installed `merge_prep.py regenerate` ran against the integrated checkout:
no ledger drift, no mission-manifest changes, local preservation `local_complete`,
no coverage problems; remote remains separate pending the explicitly authorized sync.
The final handoff is published before the resulting export so it also reaches JSONL.

Flight-board check-in measured 637 seconds and reported 4,432,294 aggregate rollout
tokens, including 4,316,800 cache reads, with `concurrent=true`; that aggregate is
not an attributable host-session token measurement. It is recorded as reported
rather than used to claim cost savings. The installed coordination command resolves
its project from its own installed path; its two new events and generated budget
projection were carried into this source-record PR before final extraction.
