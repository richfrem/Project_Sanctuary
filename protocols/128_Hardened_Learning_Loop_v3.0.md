# Protocol 128: Hardened Learning Loop & Sovereign Lifecycle (v3.1)

**Status:** RATIFIED & CANONICAL  
**Version:** 3.1 (2026-11-03; supersedes v3.0; filename kept so existing links keep working)  
**Domain:** Governance & Cognitive Continuity  
**Execution Enforcement:** `operating-sanctuary-guardian`, `preserving-soul-traces`, `managing-sovereign-memory`, `evaluating-agent-work`  
**Source of truth:** this file defines the lifecycle. `START_HERE.md`, `AGENTS.md` and the skill reference
`operating-sanctuary-guardian/references/protocol-128.md` are derived from it; if they disagree, this file wins and the others are bugs.

---

## 1. Objective

To give autonomous agent sessions an unbroken, verifiable way to inherit prior work, produce validated cognitive state deltas, and persist them, preventing "The Great Robbery" (session memory erasure) without relying on any external memory server.

```
Boot Orientation ──► Sovereign Execution ──► Retrospective ──► Technical Seal ──► Soul Persistence
 (START_HERE.md)      (APM skills, worktree)   (Layer 2 notes)   (tests + gate)     (Hugging Face)
```

---

## 2. The Five Stages

### Stage 1: Boot Orientation
Detailed in `START_HERE.md` (the 5-step bootstrap is the expanded form of this stage, not a different lifecycle).
- **Orient**: read `AGENTS.md` and `START_HERE.md` (constitutional bounds, Human Gate).
- **Wake up**: `guardian_wakeup.py --mode HOLISTIC` verifies Layer 1 health and writes `.agent/learning/guardian_boot_digest.md` (sub-second; measured ~0.1 s). It surfaces the highest-priority incomplete mission from `LEARNING/research/index.jsonl` and the **flight board**.
- **Local model gate** (when local inference is in scope): confirm the MLX server on `:8080` or run `forge_llm.py status`.
- **Hydrate Layer 1**: `context/soul.md`, `context/user.md`, `context/memory.md`, and the boot digest.
- **Progressive disclosure (Layer 2)**: use `managing-sovereign-memory` (`disclose.py --tier 1|2`). Search `LEARNING/research/index.jsonl` first (it carries each mission's title, status, verdict, `hypothesis_brief` and tags) and open a manifest only when needed.
- **Git and concurrency**: tree clean, on a `feat/*` branch, never on `main`. When other agents may be active, check `manage_coordination.py board`, then `checkout --create-worktree` an isolated worktree and lease your target files; `checkin` when done.

### Stage 2: Sovereign Execution
- **Tracks**: A (deterministic skill or script runs), B (exploratory research and doctrine work in `LEARNING/` and `protocols/`), C (micro-fix with immediate gating).
- **Skill-first**: work through installed APM skills (`plugins/sanctuary-core/` or global upstream packages). Do not invoke deprecated local scripts.
- **Human Gate Law**: any state-changing operation (writes outside scratch space, git mutations, pushes, remote uploads) needs explicit approval ("Proceed", "Go", "Execute"). "Sounds good" is not approval.
- **Popperian discipline for non-trivial work**: state a prediction and falsification criteria before running; evaluate on held-out data the work was not tuned on; report scaffold-assisted and unassisted results separately; record failures as results (a falsified hypothesis is a valid outcome and must not be forced to a pass).
- **Cost accounting**: measure expenditure with `track_budget.py` for external model dispatches.

### Stage 3: Retrospective
- **Asymmetric Persistence Rule**: when work fails, code may roll back but insights, friction points and edge cases are never rolled back.
- **Layer 2 update**: distill findings into `LEARNING/` (`concepts/`, `topics/`, `research/`), tagging confidence (`CONFIRMED`, `OBSERVED`, `HYPOTHESIS`).
- **Missions**: record outcomes with `manage_backlog.py record-result` (verdicts: CORROBORATED, FALSIFIED, ABANDONED) and run `compile`. For every mission resolved, create at least one new falsifiable inquiry (replenishment invariant). Check the shared mission IDs and chronicle numbers on `origin/main` before allocating new ones; other agents allocate concurrently.

### Stage 4: Technical Seal
- **Verify**: run the test suites of every touched skill; the pre-commit cognitive gate must pass. Audits of plugin or skill structure come from the upstream APM authoring skills when a plugin changed.
- **Pre-Completion Gate**: emit the gate defined verbatim in `AGENTS.md`: a capability check plus four numbered questions (failed or bypassed capability; guessed or corrected process; next-agent hazard; investment evaluation).
- **Score**: `score_contribution.py` for the turn's cognitive value (0-3 per dimension). Score honestly; a negative result can still score high on learning and rigor.
- **Record**: `chronicle_manager.py create` for a milestone entry; `capture_snapshot.py --type seal` for the session snapshot when a snapshot is required.
- **Commit**: stage files explicitly (never `git add -A` when other agents share the repo), commit on the `feat/*` branch, open a PR. Merge to `main` only with human approval.

### Stage 5: Soul Persistence
- **Extract**: `trace_extractor.py --source-dir chronicle/entries --append` upserts records (re-extraction refreshes every record's timestamp; only changed content is new knowledge).
- **Verify**: `trace_extractor.py --verify` checks the 13-field ADR 084 schema and SHA-256 digests; the manifest is `huggingface/metadata/manifest.json`.
- **Sync**: `sync_soul_dataset.py --dry-run` then without `--dry-run` to publish the whole `huggingface/` folder to `richfrem/Project_Sanctuary_Soul`; for a single file use `managing-huggingface-assets` (`upload_file`) and verify by re-download. Credentials come only from the shell environment; never write tokens to disk or logs.
- **Single snapshot**: `persist_soul.py` uploads one sealed snapshot (and the RLM cache, if present) through `managing-huggingface-assets`; `--dry-run` validates the skill, credentials and snapshot first. Repaired 2026-11-03 (it previously failed on a removed `huggingface-utils` plugin).

---

## 3. Memory Architecture Alignment

| Layer | Location | Governed at | Role |
| :--- | :--- | :--- | :--- |
| **1: Fast Context** | `context/` (`soul.md`, `user.md`, `memory.md`, `kernel.py`) and the boot digest | Stage 1 | Hot context loaded every session. |
| **2: Knowledge Base** | `LEARNING/` (`concepts/`, `topics/`, `research/missions/`, `research/index.jsonl`), `chronicle/` | Stages 1 and 3 | Filesystem-native, progressive disclosure, wikilink graph, 30-day confidence decay. |
| **3: Episodic Lineage** | `huggingface/` (`data/soul_traces.jsonl`, `metadata/`) and the Hugging Face dataset | Stage 5 | Immutable off-box training data that survives a total local reset. |

Note: the dataset accumulates. Records extracted from earlier repository layouts remain; see the dataset README.

---

## 4. Non-Negotiable Invariants

1. **Human Gate Law**: no state-changing operation without explicit human approval.
2. **Zero-Trust Git Discipline**: feature branches only; no commits to `main`; no `git push` without approval.
3. **Epistemic Integrity**: ground truth in tests, hashes and observed results; surface contradictions between doctrine and reality immediately.
4. **Session Persistence**: do not end a significant session without retrospective, technical seal and soul persistence.
5. **Runtime Awareness**: only installed skills exist; do not call deprecated scripts.
6. **Cognitive Investment Loop**: for non-trivial work, predict before acting, measure, score, and record the variance between prediction and result.
7. **Highest-ROI Dispatch and Replenishment**: work the highest-priority incomplete mission; each resolved mission yields at least one new falsifiable inquiry.
8. **Concurrency Safety**: when several agents share the repository, isolate work in a worktree, lease target files on the flight board, and stage only your own files.

---

## 5. Changelog

- **v3.1.1 (2026-11-03)**: `persist_soul.py` repaired and its known-gap note removed; `protocol_manager.py update` now edits the Status line in place.
- **v3.1 (2026-11-03)**: Layer 2 corrected from the removed `wiki/` to `LEARNING/`; closure gate updated to the current form (capability check + 4 questions); added local-model gate, index-first retrieval, flight board and worktree isolation, Popperian execution discipline, cost accounting, mission recording and replenishment, working soul-persistence path, and invariants 5-8; removed the claimed sub-10 ms wakeup and three audit scripts that do not exist in the repository; declared this file the single source of truth for the lifecycle.
- **v3.0**: five-stage lifecycle aligned to the 3-layer memory engine.
