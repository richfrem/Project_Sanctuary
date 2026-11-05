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
