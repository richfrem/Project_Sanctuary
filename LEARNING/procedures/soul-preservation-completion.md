# Mission completion and soul preservation integration

## Repair scope and prediction (2026-11-06)

Continue the already merged coverage/backfill repair without new missions or research.
Prediction: an atomic obligation in the final manifest, serialized local completion,
and startup retries can close the manual-capture gap without rebuilding extraction
or transport. Deterministic fixtures test this prediction; verdicts remain independent.

## Verified behavior

`record-result` saves the final disposition and obligation together. After sources are
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
