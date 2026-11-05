# LEARNING restructure (2026-11-05)

Human request: review the LEARNING folder, decide what to archive, update or leave, and whether to restructure; "you choose". Plan, in separate PRs: (1) navigation, decay and path-integrity fixes; (2) a read-based triage of the 24 `topics/` dossiers and an archive of the retired ones; (3) move the mission system (`research/` to `LEARNING/missions/` and `LEARNING/experiments/`) with every reference under `plugins/` and the core docs updated in the same PR.

## PR 1 (this one): what was wrong and what changed
- `INDEX.md` (in the guardian manifest) and `README.md` described a `wiki/` and a `missions/` folder that do not exist, called `research/` "arXiv paper analyses" and omitted the mission system and `procedures/`. Both rewritten; `INDEX.md` is now the single entry point.
- `audit_decay.py` aged undated dossiers by file mtime, which a checkout resets; none of the 10 concepts has a `Last Verified` line, so "0 decayed" was vacuous. It now uses the last git commit date (explicit `Last Verified` still wins). 5 tests, 2 failed first.
- Three concept dossiers cited `wiki/`; fixed and stamped `Last Verified: 2026-11-05`.
- A new test (`test_learning_index.py`) fails if `INDEX.md` omits a top-level entry or a concept, or if any skill, script, protocol or core doc names a `LEARNING/` path that does not exist. It found three real dangling references: `SKILL.md` -> `LEARNING/concepts/INDEX.md`, `asymmetric-persistence.md` -> `LEARNING/friction/` (the code writes to `context/memory.md` and `concepts/`), and Protocol 126 -> a dead `00_PROTOCOL` path (re-pointed to `topics/quantum_error_correction/`, its evidence base).

## A correction to my own first triage
I first sorted `topics/` from names, age and reference counts and listed 14 folders as "confident archive". Cross-checking against protocols, ADRs, context and plugins showed at least four were wrong: `quantum_error_correction` is the evidence base for PROPOSED Protocol 126; `knowledge_preservation_red_team` holds the ADR 079 draft behind the Hugging Face soul persistence; `attention_sinks_context_compression` backs a CONFIRMED insight in `context/memory.md`; `forge_v5_evolution` is a verified ADR 075 synthesis. So no topic is archived from its name: every one gets a read-based classification in PR 2.

## PR 2: read-based topic triage and archive (2026-11-05)
Method: a digest per topic (title, status/date lines, opening paragraph, who cites it) went to two models with the same rubric (KEEP / HISTORICAL / ARCHIVE / HUMAN; "if a topic is recent or cited it cannot be ARCHIVE"), and I checked every disagreement against the repository before deciding.

| | agy `gemini-3.8-flash-low` | Haiku 5.5 |
|---|---|---|
| Agreement with my final calls | 17 / 24 | 15 / 24 |
| Of my 8 ARCHIVE calls matched | 2 | 1 |
| Answered HUMAN (no call) | 1 | 6 |
| Would have archived something I kept | none | `llm_reasoning_diversity` |
| Tokens (measured) | 23,564 (1,056 out) | 57,228 (11,734 out, mostly reasoning) |
| Cost | $0.022 | $0.015 |
| Wall time | 13 s | 28 s |
| Quality (1-3, mine) | 2 | 2 |

Both under-archived and neither was usable as-is; their misses came from evidence the digest lacked (for example Protocol 131 cites the DRQ dossier). First same-task, two-model pair for MISSION-051: Haiku was cheaper in dollars but 2.2x the time and 2.4x the tokens (n = 1, so not a ratio yet).

Two "cited by" signals turned out to be loose name matches and were corrected by reading the match: the `astra_architectural_review` hit in `visualize_economy.py` is a chart label, and the `soul_persistence` hits are the concept in core docs, not the draft heuristics file. `astra_architectural_review` stays KEEP because it is recent (added 2026-11-01); `soul_persistence` is HISTORICAL.

Result: 8 topic dossiers, `security/` and two loose files (53 files, about 280 KB) moved by `git mv` to `ARCHIVE/LEARNING_pre_2026-11/` with a map and restore instructions; `research/2026-02-12/` joined `topics/` as `arxiv_2602_deep_dive`; `topics/README.md` marks the 17 remaining topics KEEP (6) or HISTORICAL (11). New guard tests: every topic has a status row, every moved item is in the archive and mapped, archived files are tracked, and no kept document links into moved paths. ARCHIVE matches an ignore rule and its existing exception is ineffective, so the move relies on `git mv` keeping files tracked.

Not done: the 134 broken relative links inside `topics/` are pasted links from external repositories (for example in `astra_automaton_opinion.md`), not internal links; left as they are.

## PR 3, step 1 of the mission-system move: `missions/` to the LEARNING root (2026-11-05)
The human's layout: `missions/`, `experiments/`, `index.jsonl`, `backlog.md` and `inbox.jsonl` move up to the LEARNING root one at a time; once `research/` is empty, the contents of `topics/` move into it and the empty `topics/` goes away. This step moved the 51 manifests and `mission.schema.json` (kept beside the manifests) from `LEARNING/research/` to `LEARNING/missions/`.
- Tests first (4 red): new location, no tracked file names the old path, the manager ignores the schema file when loading manifests, and a manifest left at the old path is reported.
- 51 files had the old path as a literal and were rewritten; 11 Python files built it piecewise and were edited. Everything under `plugins/` that named it was updated in the same PR; `chronicle/`, the Hugging Face dataset, `ARCHIVE/` and the append-only `events.jsonl` keep the old path as history.
- Bugs the tests caught on the way: the wakeup counted `mission.schema.json` as a fourth manifest (now counts `MISSION-*.json`); the backlog loader read every `*.json` in the folder (now `MISSION-*.json`); a relay fixture relied on `mkdir research/missions` creating `research/`.
- Guard: `manage_backlog.py validate` (and so the pre-commit gate) now fails if anything is left at `LEARNING/research/missions/`, so an agent following an old prompt cannot silently write there.
- Two canonical docs got path-only edits: the layer table in Protocol 128 and the architecture diagram.

### Step 2: `backlog.md` to the LEARNING root (2026-11-05)
The generated human view moved from `LEARNING/research/backlog.md` to `LEARNING/backlog.md` (manager, pre-commit hook, merge-prep and impact lists, docs, tests; the merge-harness expected-files set too). Three tests first (red), 324 pass after. No stale-write guard: the file is generated, so compile just writes it in the new place. Same history rule as step 1.

## PR 3 as shipped (2026-11-05): the whole mission-system move, then references
After step 2 the human changed the method: "do all moves first, then fix references". The branch was allowed to be broken in the middle (nothing merged until the tests passed again), which avoids rewriting the same documents four times. Final layout:

```
LEARNING/  INDEX.md README.md  backlog.md index.jsonl inbox.jsonl      (generated view, generated index, idea inbox)
           concepts/ procedures/ philosophy/                           knowledge
           missions/ (MISSION-NNN.json + mission.schema.json)   experiments/
           research/ (the 17 research dossiers; formerly topics/, with its README status table)
```
- Moves: `research/experiments`, `index.jsonl`, `inbox.jsonl` to the root; the contents of `topics/` into the emptied `research/`; the empty `topics/` removed. 110 renames, history kept by `git mv`.
- References: tests first (3 red: layout, no retired path named anywhere, manager uses the root files), then one rewrite with a fixed order (mission-system paths first, then `topics` to `research`, so the new `research/` paths were not remapped). 84 files plus the manager rewritten by hand (`research_dir` became `learning_dir`). History untouched as before: chronicle, dataset, archive, the append-only events log, and experiment records and recorded data.
- Things a blind rewrite would have broken: four experiment scripts find the repo root by counting folders up and moved one level (`parents[4]` to `parents[3]`); `.gitignore` hid a local `drq_repo` clone under the old `topics/` path; the stale-write guard now reports any mission-system item left inside `research/`; test and archive-README restore recipes used the old names.
- `procedures/` reviewed on request: `git/` deleted (superseded by `checkout --create-worktree`), `orchestration/` and `architecture/` archived (superseded by `cli-agent` and the orchestration skills). The folder stays as the home of the mechanism because three open missions (042, 047, 049) build on it and five core docs advertise it; only two outcome events were ever recorded, both on 2026-11-02.
- Answered, not changed: `philosophy/` and `concepts/` stay separate (immutable source doctrine vs confidence-tracked, script-written canon); a `concepts/philosophy/` subfolder would work but the graph and disclosure tools recurse and would start listing the 12 texts.

## Stale-reference sweep (2026-11-05, after the restructure)
Asked: do the plugin file references match the new locations, and are `context/`, `.agent/`, `INSTALL.md`, `START_HERE.md`, `AGENTS.md` and `architecture.md` current? Audited by extracting every repo path from those files and checking it exists, plus reading the claims.
- **Source `plugins/` was clean** for every moved path (the path-integrity test and an AST scan of every script's `"LEARNING" / ...` expressions). One sentence in `bootstrapping.md` still called `topics/` and `security/` the historical folders (fixed).
- **The installed copies were stale.** APM project installs are real copies in `.claude/skills`, `.agents/skills` and `apm_modules/_local/`, dated 1-2 November; they differed from the source by 13 / 21 / 9 / 2 / 3 / 1 files per skill (no `validate_missions.py`, old `LEARNING/research/...` paths in the scripts agents actually run). Fix: `apm install` at the repo root (dry run shows it touches only `./plugins/sanctuary-core`) and a new `installed-skills` line in the session wakeup that WARNs on drift (17 ms, tested).
- **`context/`:** `memory.md` named a `backlog.json` that never existed, described `LEARNING/` as "concepts/topics", said `record_insight` promotes into `LEARNING/research/` (it writes `concepts/`), and counted 6 skills (there are 7); `soul.md` had the same layer and skill-count errors and a dead path to a retired plugin. Fixed, and two durable entries added (#016 layout / board / installed copies; #017 MISSION-035 outcome).
- **`.agent/`:** `learning/README.md` listed a `learning_manifest.json` that is archived and used a `wiki/` path; `rules/coding-conventions.md` pointed at a renamed skill under `.agents/skills/` (it is `enforcing-coding-conventions`, installed globally); `frictionlogs/friction_log_verify_workflow.md` (Feb 2026) documents a retired `tools/orchestrator` and moved to `ARCHIVE/legacy_agent_learning/`.
- **The four docs:** `AGENTS.md` listed `topics/` and had no rule for leased worktrees, the untracked board, `merge_prep` or reinstalling after plugin edits (added as rule 9 and a sentence in rule 4); `architecture.md` showed `topics/` in the Layer 2 diagram; `INSTALL.md` had no "reinstall after editing" note; `README.md` linked `wiki/philosophy/...`; `START_HERE.md` was current. `INSTALL.md` also says APM >= 0.35 is needed for the `agy` target while this machine runs 0.33 (not changed; environment).
- **Found but not changed (pre-existing, not caused by the moves):** 31 plugin doc references to retired trees (`forge/`, `01_PROTOCOLS`, `mcp_servers`, retired plugin names; most `scripts/x.py` hits are skill-relative paths, not errors), 13 in canonical `protocols/` (human-gated), and `.agent/data/cortex.json`, a January 2026 record from a retired memory subsystem.
- **Guards added:** the navigation docs must have no dangling repo path, must not describe `topics/` as current, and must tell agents to reinstall; the wakeup compares installed skills with the source.

### Follow-up to the sweep: the first drift check had a false positive
After `apm install` the wakeup still reported `bootstrapping.md` as drift. Cause: APM deliberately rewrites relative links between skills on install (`../../managing-sovereign-memory/...` becomes `../../../../apm_modules/_local/sanctuary-core/.apm/skills/managing-sovereign-memory/...`). The check now maps both forms to one neutral prefix before hashing (2 tests, one red first); a real text change next to a rewritten link is still reported. `apm install` also generated a `.github/hooks/` folder for the copilot target (now gitignored like `.claude/` and `.agents/`) and refreshed the tracked `apm.lock.yaml` (deployed-file list and hashes; no machine paths or secrets). The reinstall also registered the SessionStart and PreCompact hooks in the project's `.claude/settings.json`, `.codex/hooks.json` and `.github/hooks/`, which activates the MISSION-030 snapshot hook in this project. A gitignored local `model_card.yaml` in the main checkout's forge skill is copied by the install but absent from worktrees and clones; it shows as an "extra" file only when a worktree's source is compared with the main checkout's install.

### Finding from the reinstall: installed copies would have leaked the held-out evaluation generator
After `apm install`, `test_local_agent_loop.py::test_deny_list_blocks_reads_and_hides_entries` failed in a worktree that had its own install. The local-agent sandbox (MISSION-022) hides `agent_eval_tasks.py`, the held-out task generator, from the model under test, but only by the SOURCE path prefix. APM copies each skill's whole folder, tests included, so the generator (and the guardian tests that import it) became findable and readable at three more places: `.claude/skills/`, `.agents/skills/` and `apm_modules/_local/sanctuary-core/`.
- **Past results were not exposed:** the previous lockfile shows only `test_coordination.py` from that skill's `tests/` was ever deployed, so the MISSION-022 held-out runs (3 November, recorded FALSIFIED) could not have read the generator through an install. The exposure starts with today's reinstall, which is why it matters for any future held-out run (for example MISSION-051 or 042).
- **Fix (tests first, 3 red):** the sandbox now hides by name and by folder wherever they sit, on by default: `DEFAULT_DENY_NAMES` (`agent_eval_tasks.py`, `heldout_results*.json`) and `DEFAULT_DENY_GLOBS` (`*operating-sanctuary-guardian/tests`, i.e. that folder and every installed copy of it). Other skills' tests stay visible. The old assertion was a blunt "string never appears"; it now checks that no hit's path points into a denied file, so documentation that merely names the generator is fine.
- **Lesson:** a deny-list written as path prefixes of the source tree silently stops protecting anything once the tree is copied; protect by name.

### Correction to my own lockfile commit (#338): it recorded compiled bytecode
The `apm.lock.yaml` I committed in #338 listed 36 bytecode entries (`__pycache__/*.pyc`, hashes tied to `cpython-313` and `pytest-9.1.1`): I had run the tests in that worktree before `apm install`, APM copied the `.pyc` files and recorded them, and APM merges into the existing lock so even cleaning the tree did not remove them. A lockfile that depends on which Python and pytest ran last cannot be reproduced, which is why the main checkout's own install produced a different file. Fixed: a guard test (`test_apm_lockfile_lists_no_compiled_bytecode`, red first) and a lock rebuilt from scratch from a bytecode-free tree (`find plugins -name __pycache__ -type d -prune -exec rm -rf {} +`, delete `apm.lock.yaml`, `apm install`): 0 bytecode entries, no machine paths. Recipe for next time: clean the pycache first, and install before running any tests.

### Retired-tree references: `01_PROTOCOLS` and `forge` (2026-11-05, human supplied the mapping)
`01_PROTOCOLS/` became `protocols/`; `forge/` went to `ARCHIVE/forge/` and its tooling to the `forging-sovereign-models` skill. Audit of every reference outside history first (not a blind rewrite):
- `01_PROTOCOLS/131...`, `01_PROTOCOLS/23...` mapped to `protocols/` exactly; `01_PROTOCOLS/27_The_Doctrine_of_Flawed_Winning_Grace.md` was renamed with a version suffix (`protocols/27_..._v1.2.md`). The three philosophy texts that cite it got a path-only edit.
- `forge/scripts/{fine_tune,merge_adapter,convert_to_gguf}.py` exist only as archived reference code in `ARCHIVE/forge/whole_genome_crucible/scripts/`, not in the installed skill (its `scripts/` has the dataset builder, Modelfile generator and environment check). `qlora-workflow.md` now points there and says so.
- `forge/data/train.jsonl`, `forge/data/sanctuary_genome_train.jsonl`, `forge/Modelfile` are not dead links: they are output paths the scripts create on first run. `forge/` was not gitignored, so a `git add -A` after a run could have committed generated training data; `/forge/` is now ignored (root only; `ARCHIVE/forge` is unaffected) and a test checks it.
- Guard tests (3, red first): nothing points at `01_PROTOCOLS/`; any `forge/` path is a real file or a declared output; the output workspace is gitignored.
- Not changed, with verified homes: `ADRs/084...` has a home in `ARCHIVE/ADRs/`, and the `wiki/...` paths in Astra's review dossiers map to `LEARNING/...`. Those sit in historical dossiers (records of exchanges about the `wiki/` era), `ADRs/078`, `mcp_servers/*` and `docs/*` have no home, and `docs/a*.md`-style paths are test fixtures. Only the two roots above were mapped, as instructed.

### Found before the next session: the SessionStart hook never injected the Predecessor Relay
Asked whether a fresh session has what it needs. The manual `guardian_wakeup.py` prints the Predecessor Relay (last handovers, established results, dispatchable missions, fuel gauge) and it worked (it listed this session's check-in handoffs), but `session_start_wakeup.py`, the hook a new Claude Code session actually receives, builds its own digest in-process and never called `get_relay_section`, so a fresh session got green checks and no handoff notes unless it ran the wakeup by hand. MISSION-025 integrated the relay into the wakeup; MISSION-024's hook predates it, and the only relay test covered the manual path. Fixed with tests first (red): the hook now includes the relay (5,973 chars injected, 211 ms median against the 1,500 ms ceiling) and survives a malformed events log. Also fixed: the wakeup's chronicle line counted the ledger's two keys ("2 entries") instead of its entries (now 369).
