# MISSION-060 step 2 — Haiku delegate result

Root verification: the source diff and clean companion feature worktree were
checked; all 33 tests independently rerun and PASS. Router reports exit 0 and
336.776 seconds; tokens unknown. The delegate's "no provider model calls" claim
below concerns utility/account reads and absence of nested dispatch; this Haiku
session itself is a model call. PROGRESS-HANDOFF.md was an existing file modified
by the delegate, not a newly created file. Root had already renewed the lease to
18:45:41 UTC, superseding the older lease warning below. No control-plane decision
or publication/install authorization is inferred from delegation.

Delegate: Claude CLI, Haiku 5.5, under the codex flight-board lease (not checked out, renewed or stolen).
Date: 2026-11-07, approximately 17:44–17:55 UTC. Contract: `HAIKU-STEP2-CONTRACT.md`.

**Overall status: PARTIAL.** Quota evidence refreshed; the parser defect is repaired locally in companion source (not merged, not installed). The generic control-plane question is still a human decision. Step 2 is not complete, and no study phase ran.

## 1. Quota readiness (zero-token or native no-model-call reads only)

Floor: 25% remaining per backend. All three backends are above it on these fresh reads.

| Backend | Fresh reading (UTC) | Remaining | Method | Evidence |
|---|---|---|---|---|
| Claude | session 0% used (reset Nov 7 3:39pm PT); week 65% used (reset Nov 9 8:59pm PT) | session 100%, week 35% | `claude -p /usage` (zero-token display) | `verification/haiku-quota-20261107.json` |
| Codex | 5h 12% used (reset 21:52:27 UTC); week 63% used (reset Nov 11 14:22:34 UTC) | 5h 88%, week 37% | `codex app-server` JSON-RPC `account/rateLimits/read` after `initialize` (no model call) at 17:50:08 UTC | same JSON |
| AGY | Gemini weekly 93%, five-hour 90%; Claude/GPT weekly 50%, five-hour 100%; Gemini 4 Argon weekly 100% | smallest 50% | `agy -p /usage --output-format json`, `total_tokens` reported 0 | same JSON |

What this changes:
- **Claude parser.** The live output today includes a reset suffix, so the installed parser reads it correctly (session 0, week 65). The reset-less form from the earlier gate snapshot (`Current session: 0% used`) is still unparsed by the installed code (`session_pct_used=None`). Richard's defect report stands for that form. It is repaired in companion source, below. Until the companion change is merged and installed, automated gate checks still read the reset-less form as unparsed.
- **Codex staleness.** The 07:14 UTC rollout is superseded by the 17:50 UTC app-server read. Note the boundary: the previous 5-hour window reset at 17:48:35 UTC. A read at 17:48 showed the old 10%-used window; the 17:50:08 read is the rolled-over window. The 5-hour value is fresh as of 17:50, but any later use should re-read it.
- **Claude sign-in.** The CLI warns the sign-in ends between 2026-11-10 and 2026-11-13. This needs a human `claude auth login` or `setup-token`. I did not run it.

## 2. Control-plane applicability and configuration

Read-only status (`control_plane_mode.py status`): **ENABLED (no mode file: default)**, **INCONSISTENT**. Neither guard is installed or wired in this clone's `.git/hooks`. `context/control_plane.db` is absent. AGENTS Phase 0 section absent. Installation probe reports FRESH (uninitialized, not a failed approval). No mode change, init, retrofit, key or signing action was taken.

Evidence on applicability:
- `running-control-plane/references/modes/mode.md` (the mode reference): a missing mode file means `enabled`, and enable installs and wires guards before declaring enabled. The mode is therefore a default, not a recorded choice, and it is not operating because the guards are absent.
- Sanctuary's adapter, `plugins/sanctuary-core/.apm/skills/evaluating-agent-work/references/shared-loop-adapter.md` (line 7): the policies that govern dispatch and merge are the flight-board lease with a worktree before dispatch, Human Gate (`authorization.may_merge` defaults false), and headroom. The adapter does not name the generic control-plane state machine as a precondition for research execution.
- Constitution Supreme Law: no state-changing operation without explicit human approval. This governs the pending mode change, and it is unresolved.

Conclusion: the generic plane is declared enabled but does not operate, and I can't call its gate passed or failed. Whether it must apply to this research mission is the open human decision (disable the unused plane vs keep enabled and prepare a compatible setup). Full retrofit remains blocked, since it would create top-level `wiki/` and `references/`, which conflicts with the canonical domain rules. No human signing identity is enrolled (gates-20261107.md, not re-run here, since signing checks were not needed to decide this). I found no evidence in the adapter or constitution that a signature is required for the next research phase. That is my inference and the human should confirm it.

## 3. Repairs

**Companion source (local only).** Repo `richfrem/agent-plugins-skills`, branch `feat/usage-parser-resetless`, worktree `/Users/richardfremmerlid/Projects/agent-plugins-skills/.worktrees/usage-parser-resetless`, base `main` at `7a4c29eb`, commit `c93f4b7b`. Not pushed, not merged, not installed. Changed paths:
- `plugins/agent-orchestration/.apm/skills/evaluating-agent-outcomes/scripts/usage_measure.py`: the reset suffix is optional for both windows. A displayed percentage is kept. An absent reset becomes `None`, not zero or an imputed value.
- `.../tests/test_scoring_accounting.py`: new regression test `test_resetless_session_is_parsed_as_displayed_not_unparsed`.
- `.../tests/fixtures/claude_usage_resetless.txt`: new fixture.
- `plugins/agent-orchestration/references/evolution-log.md`: `DEBT-20261107-USAGE-RESETLESS-SESSION (RESOLVED)` row, as Protocol 133 section 4 step 5 requires for `plugins/` commits.

Patch exported for review: `LEARNING/experiments/mission-060/verification/haiku-usage-parser-fix.patch` (99 lines).

**Sanctuary.** New files only: `HAIKU-STEP2-RESULT.md`, `verification/haiku-quota-20261107.json`, `verification/haiku-usage-parser-fix.patch`, plus the progress update. No existing tracked file was edited for this delegate's work.

## 4. Checks and results

| Check | Result |
|---|---|
| Companion baseline tests before change (`unittest discover`, evaluating-agent-outcomes) | 32 OK |
| Companion tests after change | 33 OK |
| `py_compile` on the changed `usage_measure.py` | OK |
| `audit_skill.py <skill> --mode source` (Protocol 133 step 4) | PASS |
| Installed parser on live Claude output | parsed correctly |
| Installed parser on reset-less text | `session_pct_used=None` (defect, pre-fix) |
| Repaired parser on reset-less fixture | session 0, reset None, week 65; absent session stays None |
| Installed copies in `~/.claude`, `~/.agents`, `~/.gemini` | Read-only `cmp` against companion `main`: all three match the old parser. Not edited. |

## 5. Failures and recovery (recorded, not erased)

- My first `Edit` on the companion parser failed: the file had not been read in this session. Read it and retried. No file was changed by the failed attempt.
- My first attempt at the test insertion was a no-op edit (old and new strings identical). Corrected with a real insertion.
- My first evolution-log row had five cells, not six. Fixed to match the table header. Verified by cell count.
- The pre-commit guard in the companion clone printed `no task registered for branch ... commit is ungated`. This is a warning, not a block. The commit proceeded with the staged paths only.

## 6. Usage and cost

- Model-token usage for this delegate session: **unknown** (not reported to me by the CLI). I made no provider model calls: `claude -p /usage` and `agy -p /usage` are zero-token display commands, and the Codex read is a native JSON-RPC account read.
- Claude `/usage` and Codex `account/rateLimits/read` token counts were not independently metered. The AGY `total_tokens: 0` is the CLI's own report.

## 7. Lease, worktree and boundaries

- Sanctuary worktree `/Users/richardfremmerlid/Projects/Project_Sanctuary/.worktrees/codex-mission-060`, branch `feat/codex-mission-060`, HEAD `f469f62ba`. Lease owner `codex`. I did not checkout, checkin or renew the lease. The gates file recorded a renewal to 17:54:31 UTC, which is very close. **The root agent must heartbeat before any further mission work.** I did not re-verify the board.
- No push, merge, PR, global install, installed-copy edit, key or signing action, `allowed_signers` change, mode change, calibration, freeze or held-out run.

## 8. Remaining decisions and next actions

1. **Richard (human):** choose disable-unused vs keep-enabled-and-prepare-compatible setup for the generic control plane. Until chosen, the applicability gate stays UNRESOLVED, not passed.
2. **Richard (human):** decide on the companion parser fix. Next steps, each needing explicit authorization: push and open a PR for `feat/usage-parser-resetless`; squash-merge on go-ahead; `apm update --global --dry-run`, then `--yes`; verify installed copies against merged `main` (`diff -rq`). After this, the automated gate reads the reset-less form.
3. **Richard (human):** renew the Claude sign-in before 2026-11-10.
4. **Root agent:** heartbeat the codex lease. Recheck quota before any calibration, since the Codex 5-hour window changes within hours.
5. Do not repeat: a `claude -p` model probe to change the usage display, or any treatment of unknown quota as zero.
