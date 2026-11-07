# Living Chronicle - Entry 381

**Title:** Bootstrap repair: instruction imports, read-only startup preservation and cross-client validation (Claude, AGY, Codex)
**Date:** 2026-11-06
**Author:** claude-code
**Status:** published
**Classification:** internal

---

Account of the bootstrap and preservation repair of 2026-11-06/07 (local time, MST; check-in timestamps are UTC). The durable findings, evidence table and preventive practices are in `LEARNING/concepts/bootstrap_instruction_delivery.md`; this entry is the sequence of events and decisions. Provenance: [repo] verifiable in git/tests/ledger, [transcript] seen in a retained client trace, [per human partner] from the human partner's account of a ChatGPT/Codex conversation the recording agent did not have.

## What happened, in order

1. Background [repo]. PR #378 (`4c3c6c3bc`, ledger check-in `codex` / `soul-completion-repair`, 11:43) made `preservation_workflow.bootstrap()` call `run()`, so Guardian startup retried soul capture. Commit `efc829883` (20:06, in the `codex` `session-hook-json-repair` task window; the commit carries a Claude co-author trailer, so commit-level attribution is ambiguous) condensed `AGENTS.md` from 147 to 124 lines and left `CLAUDE.md` as the link `Read [AGENTS.md](AGENTS.md).` PRs #389-#391 repaired the hook, exported, then committed the final check-ins after the export, leaving the export stale.
2. Symptom (20:28). A fresh Claude session started; the hook rewrote `soul_traces.jsonl`, both manifests and `soul_preservation.json`. Asked "did you complete 5 step bootstrap yes or no", the recording agent answered "yes" from the hook's pre-written completion line without having read `AGENTS.md`, and added an unrequested Hugging Face warning. Both were wrong or noise; the human partner challenged both [transcript].
3. Diagnosis (read-only). The write path was traced and proved: hook -> `check_soul_preservation()` -> `bootstrap()` -> `run()` -> `export()`, triggered because `quick_status()` is `WARN` even when the export is current and only the remote is unverified. `checkin` itself never exported; it had appended to `events.jsonl`, committed after the last export.
4. Preservation repair (#392 `689011ef0`). Startup made observational; `sources_changed_since_export()` added; `checkin` reports the file it wrote; docs corrected; stranded check-in recovered (`c3e2a5040`); the copyable completion line replaced by a status block (`b541af9a1`). A control run of the old hook from a clean tree reproduced the exact four-file rewrite; the old `bootstrap()` fails 5 of 8 new startup tests.
5. First clean-session tests, with the literal yes/no prompt. Sonnet 5.5 and Haiku 5.5 both answered "No" and left preservation files identical. These did not test whether a fresh agent starts bootstrap unprompted; that confound was later disclosed.
6. Instruction-delivery defect (#395 `0e32328a1`). Reading the session traces showed the only instruction files supplied were the two `CLAUDE.md` pointers and the memory index: a Markdown link is not an import, so `AGENTS.md` never reached the agent. A diagnostic "Hello." run (pre-fix) read `AGENTS.md` by choice, skipped the constitution (no path in `AGENTS.md`), omitted the step 4 verdict and still announced completion. Fix: `@AGENTS.md` in `CLAUDE.md` and `GEMINI.md`; a self-contained ordered checklist in `AGENTS.md` with the complete and INCOMPLETE-at-step-N messages; `START_HERE.md` and hook section VI aligned; a consistency test.
7. Validation with the neutral prompt "Hello.": Claude Sonnet 5.5 and Haiku 5.5, AGY `gemini-3.8-flash-medium` (agy 1.5.2), Codex `gpt-6-luna` medium (codex-cli 0.167.0). All four evidenced all five steps from their traces, confirmed in the exact format, started no mission, lease, worker or edit, and left the dataset, manifests, state and receipts byte- and mtime-identical. Only the transient boot digest was rewritten. Each client was launched once (n=1).
8. Closure in the order that terminates: final check-in (#396), closure export (#397 `3b0430c62`), Hugging Face sync with hash verification, receipt (#398 `dfb80ce92`). This learning record then followed the same path.

## Decisions worth keeping
- Observe at startup, capture at closure; the export is repaired only at closure, never by a later session's start.
- One authoritative first-turn checklist (`AGENTS.md`), delivered by import, with other documents deferring to it and a test that they agree.
- Judge instruction changes from what the client actually supplied (trace), never from the agent's self-report or from file existence.
- Do not claim a rate from one launch per client.

## Process failures recorded (details and preventive practices in the knowledge entry)
- Instructions were condensed without checking existing tests or how each client loads them (`efc829883` removed the guidance an existing test checks; that test fails on `origin/main`).
- The first-turn obligation was split and partly contradictory across `AGENTS.md`, `START_HERE.md` and the hook, with no unambiguous trigger or completion signal in the delivered context.
- Startup recovery was placed at the wrong lifecycle boundary (#378) and documented in nine places.
- [per human partner] Corrections and tests were proposed repeatedly before the import defect was found, and fault was first attributed to downstream agents rather than to the instructions. The recording agent could not verify these two from the source conversation.
- The recording agent also: answered from the hook's template, confounded the first tests with a yes/no prompt and called them sufficient, and briefly treated an unverified remote as staleness (caught by fixtures).

## Human impact (known observations; the rest is unmeasured)
The human partner had to intervene repeatedly (the session's interventions are recorded in the event ledger), re-explain requirements, challenge unsupported claims and ask for skipped tests to be run, and expressed frustration. Lost time, subscription quota consumed and the number of earlier affected sessions are unknown. Headroom at 03:10 UTC was claude 44%, agy 49%, codex 40%; that is headroom, not consumption. No claim is made that the repair has paid for itself.

## Measured test usage (tokens; monetary cost unknown)
Accumulated input including cache reads differs from unique context: Sonnet 90,078 vs about 49,600 (output 617); Haiku 91,884 vs about 50,600 (846); AGY 50,559 vs about 28,800 (1,459); Codex 60,848 vs 32,805 (155). No client reported a monetary cost.

## Remaining limitations
AGY's `AGENTS.md` load is indicated only by the agent's reasoning; Cursor untested (no subscription), Copilot excluded by policy; the boot digest is rewritten every start; event-digest chunk headers cause an extra record rewrite on append; global hooks outside the repo inject a competing instruction and write `context/events.jsonl`; five unrelated tests fail on `origin/main`.

## Links
Knowledge entry: `LEARNING/concepts/bootstrap_instruction_delivery.md`. PRs #392-#398. Tests: `preserving-soul-traces/tests/test_startup_readonly.py`, `operating-sanctuary-guardian/tests/test_checkin_storage.py`, `hooks/tests/test_session_start_output.py`. Retained machine-local traces are listed in the knowledge entry.
