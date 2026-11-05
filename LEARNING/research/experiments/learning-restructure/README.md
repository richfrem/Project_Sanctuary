# LEARNING restructure (2026-11-05)

Human request: review the LEARNING folder, decide what to archive, update or leave, and whether to restructure; "you choose". Plan, in separate PRs: (1) navigation, decay and path-integrity fixes; (2) a read-based triage of the 24 `topics/` dossiers and an archive of the retired ones; (3) move the mission system (`research/` to `LEARNING/missions/` and `LEARNING/experiments/`) with every reference under `plugins/` and the core docs updated in the same PR.

## PR 1 (this one): what was wrong and what changed
- `INDEX.md` (in the guardian manifest) and `README.md` described a `wiki/` and a `missions/` folder that do not exist, called `research/` "arXiv paper analyses" and omitted the mission system and `procedures/`. Both rewritten; `INDEX.md` is now the single entry point.
- `audit_decay.py` aged undated dossiers by file mtime, which a checkout resets; none of the 10 concepts has a `Last Verified` line, so "0 decayed" was vacuous. It now uses the last git commit date (explicit `Last Verified` still wins). 5 tests, 2 failed first.
- Three concept dossiers cited `wiki/`; fixed and stamped `Last Verified: 2026-11-05`.
- A new test (`test_learning_index.py`) fails if `INDEX.md` omits a top-level entry or a concept, or if any skill, script, protocol or core doc names a `LEARNING/` path that does not exist. It found three real dangling references: `SKILL.md` -> `LEARNING/concepts/INDEX.md`, `asymmetric-persistence.md` -> `LEARNING/friction/` (the code writes to `context/memory.md` and `concepts/`), and Protocol 126 -> a dead `00_PROTOCOL` path (re-pointed to `topics/quantum_error_correction/`, its evidence base).

## A correction to my own first triage
I first sorted `topics/` from names, age and reference counts and listed 14 folders as "confident archive". Cross-checking against protocols, ADRs, context and plugins showed at least four were wrong: `quantum_error_correction` is the evidence base for PROPOSED Protocol 126; `knowledge_preservation_red_team` holds the ADR 079 draft behind the Hugging Face soul persistence; `attention_sinks_context_compression` backs a CONFIRMED insight in `context/memory.md`; `forge_v5_evolution` is a verified ADR 075 synthesis. So no topic is archived from its name: every one gets a read-based classification in PR 2.
