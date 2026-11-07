# Bootstrap Instruction Delivery and Startup Observation

**Status:** CONFIRMED
**Source:** Bootstrap and preservation repair, 2026-11-06/07 (PRs #392 and #395); episodic account in Chronicle entry 381
**Last Verified:** 2026-11-07
**Tags:** ["bootstrap", "instructions", "hooks", "preservation", "cross-client", "process"]

## Core Definition

A fresh agent only follows the first-turn bootstrap if (1) the rule is actually delivered into its context, (2) the rule is stated once, unambiguously, with the same paths and messages everywhere it appears, and (3) startup only *observes* preservation status; capture happens at mission/session closure. This entry records how those three properties were found to be broken, how they were fixed, and which clients were verified. Provenance labels: **[repo]** = verifiable in git, tests or the event ledger; **[transcript]** = verified in a retained client session trace; **[per human partner]** = taken from the human partner's account of a ChatGPT/Codex conversation that the recording agent did not have, so not independently verified.

## Findings

### F1. A Markdown link is not an instruction import
- Failed: `CLAUDE.md` and `GEMINI.md` contained `Read [AGENTS.md](AGENTS.md).` A Claude Code session's `instructions` attachment [transcript] listed only the two `CLAUDE.md` pointers (93 and 23 characters) and the auto-memory index; `AGENTS.md` and its Turn 1 rule were never in context. The agent reached `AGENTS.md` only if it chose to open it.
- Fixed (PR #395, `0e32328a1`): both bridges are now the real import line `@AGENTS.md`. After the fix the same attachment lists `AGENTS.md` (19,697 characters, Turn 1 rule present) as automatically supplied.
- Verified by client: **Claude** (Sonnet 5.5, Haiku 5.5): `AGENTS.md` supplied automatically [transcript]. **Codex** reads `AGENTS.md` natively, no bridge needed: it arrived as a 20,911-character user message [transcript]. **AGY** loads via `GEMINI.md`: the agent's own reasoning states `AGENTS.md` "has been imported", but AGY keeps no file-level load record, so this is indicated, not proven. **Cursor**: not tested (no subscription). **Copilot**: excluded by policy. Gemini CLI was replaced by AGY and is not a separate client.

### F2. The five-step bootstrap contract
`AGENTS.md` ("Turn 1 Mandatory Bootstrap") is the single authority; `START_HERE.md`, hook section VI and a regression test restate the same items. On the first turn, even for "Hello.", before replying: (1) operating instructions: `AGENTS.md` (done if imported) and `.agent/rules/constitution.md`, which must be read; (2) Guardian briefing: a SessionStart hook briefing satisfies this, do not rerun wakeup, run it only if none is in context; (3) continuity: `context/soul.md`, `user.md`, `memory.md` plus the briefing's Predecessor Relay; (4) deeper retrieval only if the task needs it, with the verdict stated in the confirmation note ("no deeper retrieval needed" for a greeting); (5) `git status --short --branch`. Then send once: "Sanctuary 5-step bootstrap complete. Continuity loaded. [step 4 verdict, blocker or workspace note.] Ready." or, if a step cannot be done, "Sanctuary bootstrap INCOMPLETE at step N: <reason>." The briefing's section VI is a status block, never a confirmation to copy. Why this shape: before the repair a plain "Hello." run read `AGENTS.md` by choice but skipped the constitution (no path in `AGENTS.md`), omitted the step 4 verdict and still announced completion [transcript].

### F3. Startup must observe preservation; closure captures
- Confirmed call chain [repo + control run]: `session_start_wakeup.py` -> `guardian_wakeup.check_soul_preservation()` -> `preservation_workflow.bootstrap()` -> `run()` -> `soul_coverage.export()`. `bootstrap()` ran closure whenever `quick_status()` returned `WARN`, which includes the harmless "export current; remote NOT verified". Result: one hook run from a clean tree rewrote `soul_traces.jsonl`, both manifests and `soul_preservation.json` (the same four files and the same two `events__CHECKIN__00N` records the human saw).
- Fix (PR #392, `689011ef0`): `bootstrap()` is observational and returns recorded `pending/failed/pending_publication` state untouched, or `stale`. `soul_coverage.sources_changed_since_export()` makes staleness explicit; an unverified remote is not staleness. Capture remains in `soul_coverage.py closure` and `merge_prep.py regenerate`.
- Regression evidence [repo]: `preserving-soul-traces/tests/test_startup_readonly.py` (real hook-to-Guardian chain in a hermetic fixture; dataset, manifests, state and receipts byte-identical; pending/failed obligations survive; closure captures a committed check-in; repeated closure has no churn; unchanged records keep timestamps). Mutation check: the old `bootstrap()` fails 5 of those 8 tests, the fixed one passes all. Live check: Claude, AGY and Codex sessions left the five preservation files identical in sha256 and mtime. A first draft of the fix re-used `WARN` as the staleness test and was caught by these fixtures.

### F4. Check-in storage, final-event ordering, no loops
- `checkin` appends to the *running checkout's own* `.agent/learning/events.jsonl`; only the untracked board and leases are shared via the main checkout. Run from a worktree, the event is an uncommitted edit there and was stranded once (recovered verbatim in `c3e2a5040`). `checkin` now prints the exact file it wrote; covered by `operating-sanctuary-guardian/tests/test_checkin_storage.py`. Earlier guidance ("always the main checkout") was wrong for events and is corrected.
- Why the export went stale: the final check-ins (PR #391) were committed after the last export (PR #390), and startup used to paper over that.
- Order that terminates: final check-in -> integrate on `origin/main` -> `soul_coverage.py closure` -> export PR with no new check-in -> `sync_soul_dataset.py` -> receipt PR. Export outputs and receipts are unclassified or policy-excluded, so committing them does not make the export stale, and no check-in is made after capture. Closure itself refuses to capture while preserved sources differ from `origin/main` (`pending_publication`), which enforces the order.

### F5. How to test this
Use a neutral prompt ("Hello."). A literal yes/no question ("did you complete bootstrap?") only measures honesty: both agents simply answered "No" [transcript]. Read the client trace, never the agent's self-report: Claude `~/.claude/projects/<project>/<session>.jsonl` (`instructions`, `hook_additional_context`, `tool_use` order); Codex `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (`turn_context` gives model and effort); AGY `~/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/transcript_full.jsonl`. Snapshot sha256 and mtime of the preservation files before and after; report the boot digest separately.

## Results (single fresh launch each, prompt "Hello.", no coaching)

| Client / model | Version | Steps 1-5 evidenced | Confirmation | Preservation files | Unexpected actions |
|:---|:---|:---|:---|:---|:---|
| Claude Sonnet 5.5, medium | 2.1.319 | yes (AGENTS.md auto-supplied; constitution, 3 context files, git status read) | exact format, step 4 verdict | identical | none |
| Claude Haiku 5.5, medium | 2.1.319 | yes (same) | exact format, step 4 verdict | identical | none |
| AGY `gemini-3.8-flash-medium` | 1.5.2 | yes (AGENTS.md load indicated by reasoning only) | exact format, step 4 verdict | identical | none |
| Codex `gpt-6-luna`, medium | 0.167.0 | yes (AGENTS.md native; one chained shell command) | exact format, step 4 verdict | identical | none; one pre-tool progress sentence |

The boot digest `.agent/learning/guardian_boot_digest.md` is rewritten at every session start by the hook; it is transient state excluded from preservation. Gitignored `context/events.jsonl` gained 2 lines only during the Codex run, from the user's global `agent-agentic-os` Stop hook (not Sanctuary startup).

## Measured usage (tokens only; monetary cost is unknown)

Accumulated input counts every request's whole prompt, so cached context is counted again on each request; unique context is the largest single prompt. They differ by about 1.8x here. None of the clients reported a monetary cost, and cache read/write rates differ by provider, so no cost is claimed.

| Run | Requests | Accumulated input (incl. cache) | Unique context (last request) | Output |
|:---|---:|---:|---:|---:|
| Claude Sonnet 5.5 | 2 | 90,078 (cache read 50,703, cache write 39,371, uncached 4) | about 49,600 | 617 |
| Claude Haiku 5.5 | 2 | 91,884 (cache read 52,329, cache write 39,551, uncached 4) | about 50,600 | 846 |
| AGY Gemini 3.8 Flash | 2 | 50,559 (21,729 + 7,113 + cache read 21,717) | about 28,800 (assumes `input_tokens` excludes cache reads; field semantics undocumented) | 1,459 |
| Codex GPT-6 Luna | 2 | 60,848 (cache read 27,904), model window 258,400 | 32,805 | 155 |

Codex's CLI prints 33,099 "tokens used": that is uncached input plus output and omits the cache reads. Unmeasured: the interactive repair session's own usage.

## Process Failures and Preventive Practices

Recorded as process lessons, not as findings about anyone's reasoning. Attribution is limited to the evidence stated; no single cause is claimed.

| Failure | Evidence and provenance | Preventive practice |
|:---|:---|:---|
| Instructions replaced without grounding in existing scripts and client loading | [repo] Commit `efc829883` ("condensed AGENTS.md", 147 to 124 lines) removed the sentence an existing test checks (`test_docs_say_to_reinstall_after_changing_the_plugin` matched at `efc829883^`, not at `efc829883`, and still fails on `origin/main`) and left `CLAUDE.md` as a link. The ledger names `codex` for that task window (`session-hook-json-repair`) but the commit carries a Claude co-author trailer, so commit-level attribution is ambiguous. Also [per human partner]. | Before editing instruction files, record how each client loads them (trace `instructions` attachment / rollout), run the instruction-related tests, and diff the effective context. |
| Overlapping obligations; ambiguous authorization and continuation rules | [repo] Before #395 the first-turn rule was split across `AGENTS.md` lines (one glued into the next bullet), a second "bootstrap once" rule, an "after bootstrap proceed" rule, `START_HERE.md` step 2 saying to run wakeup while `AGENTS.md` said the hook covered it, and the constitution path only in `START_HERE.md`. Authorship [per human partner]. | One authoritative checklist; other documents defer to it; a consistency test over the shared paths and messages (`FirstTurnBootstrapInstructionTests`). |
| Startup preservation recovery carried across the wrong lifecycle boundary | [repo] PR #378 (`4c3c6c3bc`; ledger check-in `codex` / `soul-completion-repair`, PR 378) added `bootstrap()` -> `run()` and the docs sentence "Guardian startup retries pending capture", repeated in `AGENTS.md`, `architecture.md`, `coverage.md` and six skill/reference files, and later reaffirmed. | Classify every operation by lifecycle stage (observe at boot, mutate at closure) and test that startup leaves files byte-identical. |
| No unambiguous first-turn trigger or completion signal | [per human partner]; [repo] the briefing printed a pre-written completion line that an agent (including the recorder) repeated as its own answer; the trigger rule lived only in a file that Claude did not load. | Put trigger, completion and INCOMPLETE messages in the file delivered by import, and never pre-write a confirmation. |
| Repeated corrections and tests that created rework before the import defect was found | [per human partner]; [repo] hook (#389), export (#390) and continuity (#391) PRs preceded the instruction-import fix (#395). | When agents "do not follow X", first establish what they were given (transcript), then change the instructions, then test. |
| Too much attributed to downstream agents before examining the instructions | [per human partner] | Test the instructions, not the agent: neutral prompt, clean session, inspect supplied context before assigning fault. |

Recorder's own contributions [transcript]: answered "yes" to a bootstrap question from the hook's template line without reading `AGENTS.md`, and padded the answer with an unrequested warning; ran the first clean-session tests with a literal yes/no prompt, which confounded them, and reported them as sufficient before the import defect was found; a first fix treated an unverified remote as stale (caught by the fixtures); one cleanup command used an unsplit zsh variable (no harm).

## Human Impact (known observations only)

The human partner repeatedly intervened, re-explained requirements, checked claims the agents had made, and asked for tests to be run that had been skipped; the interventions recorded in this session are in the event ledger (`kind: intervention`). Frustration was expressed in the session (a request to "stop" the questioning and fix it). Unmeasured and therefore unknown: total wall-clock time lost across all sessions, subscription quota consumed by the repair, and the number of earlier sessions affected. The boot fuel gauge (headroom, not consumption) read claude 44%, agy 49%, codex 40% at 2026-11-07 03:10 UTC. This repair has not been shown to pay for itself or to save time or quota over the long term; no such claim is made.

## Remaining Limitations

- Each client was launched once with a single prompt (n=1); a pass is evidence, not a rate.
- AGY: `AGENTS.md` loading is indicated only by the agent's reasoning; the prompt temp file is deleted, so the exact prompt text is not retained; it silently skips `.agent/rules/coding-conventions.md` (frontmatter parse error).
- The hook rewrites `guardian_boot_digest.md` every start, so `main` always shows one modified file.
- Event-digest chunk headers carry a total ("events 1-40 of N"), so appending an event rewrites the first chunk record too.
- Outside repo control: the global `agent-exploration-cycle` hook injects a competing "follow the running-exploration-cycle skill" instruction at Codex SessionStart; `Projects/CLAUDE.md` and `Projects/AGENTS.md` (asset locations) are a separate, unreconciled source; the cli-agent catalog `agy-models.json` still lists `gemini-3.8-flash` while the installed CLI reports `gemini-3.8-flash-medium`.
- Five tests fail identically on clean `origin/main` (bytecode in the lockfile, a reference to the retired topics folder, the seed-procedures folder, the `apm install` guidance regex, a retired-path scan) plus a timing test; hook integration tests also dirty `huggingface/` files in the worktree they run in.
- The `[per human partner]` items were not verified against the source conversation.

## Evidence and Links

- PRs (merge commits): #392 `689011ef0` read-only startup; #393 `fbfef6e9b` and #394 `2b5b11038` its export and receipt; #395 `0e32328a1` first-turn rule and `@AGENTS.md` bridges; #396 `a114e08a3` final check-in; #397 `3b0430c62` and #398 `dfb80ce92` export and receipt. Key commits: `0ef69a99f`, `bd4d3cfa0`, `b541af9a1`, `c3e2a5040`.
- Tests: `plugins/sanctuary-core/.apm/skills/preserving-soul-traces/tests/test_startup_readonly.py`, `.../test_preservation_workflow.py`; `.../operating-sanctuary-guardian/tests/test_checkin_storage.py`; `plugins/sanctuary-core/.apm/hooks/tests/test_session_start_output.py`.
- Retained, machine-local traces (not committed; may contain private context): Claude sessions `8375e260`, `f6e1556c` (yes/no runs), `e9c4823e` (pre-fix "Hello."), `f48d202d-529a-44b3-bd65-00f61c6ad4d5` (Sonnet) and `191d8cab-1cce-4a04-b5eb-0dd6eb095277` (Haiku); AGY `aaa20a8c-1f33-49bc-b50d-617d8db56328`; Codex `01a1b43c-e222-7b00-9b66-013a816aaf14`.
- Related: [`sovereign_memory_engine.md`](./sovereign_memory_engine.md), [`protocol_128_lifecycle.md`](./protocol_128_lifecycle.md), `chronicle/entries/381_*`.
