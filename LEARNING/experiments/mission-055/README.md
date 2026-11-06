# MISSION-055: reusable agent learning loop (closeout of the engineering work, 2026-11-06)

Ownership and scope map: [`ownership.md`](./ownership.md). No empirical verdict is recorded: this mission's claim is an integration claim, and 053/054's behavioral and longitudinal claims stay UNASSESSED in their own manifests.

## Implemented and verified (deterministic, by tests; mocked backend where stated)
- Companion PR richfrem/agent-plugins-skills#749 (merged): `evaluating-agent-outcomes` (19+4 tests), `cli-agent/run_agent.py` launch controls (`--cwd/--allowed-root`, `--claim-dir/--unit/--owner`, `--timeout`, `--clean-context`, `--result-json`; 12 new tests incl. SIGTERM claim release and crash-recorded-as-failed), `dispatch_guard.py` moved to `cli-agent` (shim left in the orchestrator), `orchestrating-agent-work/scripts/work_loop.py` and `references/learning-loop.md`.
- Two-task loop fixture (MOCKED worker executable, REAL `run_agent.py` subprocesses, non-Sanctuary repo): injected worker-PATH failure refused before launch, incomplete delivery returned for repair (worker exit is not completion), accepted on criterion-level findings, lesson recorded, retrieved BEFORE dispatch on task 2 and applied (no refusal), use and recurrence recorded, all records inside the repo root, usage unknown not zero.
- Plain task without scores; PR-style rubric; a 3/3 score cannot conceal an unmet required criterion; unevaluable failure conditions block completion; `may_merge` must be a boolean; unknown telemetry stays null; duplicate unit never launched; path-traversal ids rejected.
- Sanctuary (this PR): `eval_gate.py` is a delegating compatibility adapter (existing 19 tests unchanged and passing), `sanctuary_adapter.py` records delivery separately from the manifest's empirical state and has no verdict-writing path (tests assert it), docs in both repos, `AGENTS.md` states the two-repository responsibility.
- Independent same-vendor review of the companion PR found 9 defects (all reproduced, all fixed with tests); the Sanctuary adapter had one more defect found by running it through the installed interface (relative store path doubled), fixed with a test.
- Installs: global `apm update --global` (7 packages): installed `orchestrating-agent-work`, `cli-agent`, `evaluating-agent-outcomes` are byte-identical to the merged clone in `~/.claude`, `~/.agents` and `~/.gemini/antigravity-cli`. Local `apm install` refreshed the project copies; no project-local copy of a companion skill shadows the global one. The interrupted-install backup (`~/.apm/apm_modules/.apm-resolution-staging/fd27507e7ca8`, dated 2026-10-31) contained only an empty `replacements/` directory and no running APM process existed; it was removed with `rmdir` (no recursive delete).

## Actual agent behavior observed (one real run, installed interface)
Neutral repo, installed `work_loop.py` -> installed `run_agent.py` -> real `claude` (Haiku 5.5, 6.5 s): created `hello.py`; cwd boundary and unit claim enforced and recorded; criterion passed; recommendation accept; `integration_allowed` false (authorization not widened). Findings: the CLI printed no token usage, so usage is recorded as unknown (not captured for `claude -p` text output); and a non-clean `claude` launched in the neutral repo still loaded user hooks (a `context/` directory appeared), which is the already-known ambient-context leak. `--clean-context` exists for experiments and was not used here because this is ordinary work.

## Ownership split follow-up (same mission, 2026-11-06)
Generic scoring (`scoring.py`), cost tiers and ledger arithmetic (`accounting.py`) and usage measurement (`usage_measure.py`) now exist once upstream (companion PR #751); Sanctuary's `score_contribution.py`, `track_budget.py`, `evaluate_dispatch.py` and `usage_probe.py` keep their command lines, output and storage and call them. The Sanctuary rubric is data (`references/contribution-profile.json`); budget profiles, headroom floors, the falsified predictor, dashboards and mission records stay local. Proof of unchanged behavior: golden tests against pre-migration outputs (`tests/test_pre_migration_goldens.py`). Map: `evaluating-agent-work/references/responsibility-map.md`.
**Not complete:** `counterexample_eval.py`'s generic search core is still implemented only in Sanctuary because MISSION-041's frozen record pins its hash; extracting it requires a re-freeze decision (reported as a concrete incompatibility, not silently changed).

## Unassessed
- Whether orchestrators behave better (053 scenario families, baseline versus revised): not run.
- 054 Claim A (12 held-out scenarios) and Claim B (12 real work units, recurrence and overhead): not run / unassessed. Passive collection: `eval_gate.py repair applied`, `lessons.py use`.
- Any operational saving, quota reduction or reduction in human corrections. One demonstration proves wiring only.

## Limitations
- Controls bind only launches passing the flags; direct CLI use bypasses them. Claims are file-based (no crash-safe fencing; 038 unimplemented); a hard-killed wrapper (SIGKILL) can leave a stale claim.
- `--clean-context` is requested, not probed; usage capture works only for backends that print it.
- The 46 failures in the companion plugin suite, when run from a nested worktree, are identical on `main` and unrelated.
- Cross-vendor review not obtained (Codex quota); reviews were same-vendor.
- 004 stays open with no verdict; its readiness list is in `evaluating-agent-work/references/eval-gate.md`.
