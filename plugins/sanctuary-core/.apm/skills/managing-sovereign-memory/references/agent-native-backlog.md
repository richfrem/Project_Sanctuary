# Agent-Native Cognitive Backlog Architecture

## 1. Core Purpose & Philosophy

Project Sanctuary operates as a sovereign engineering habitat and compounding epistemic platform for autonomous AI agents. Unlike human-centric project trackers that optimize for graphical user interfaces, rich formatting, and verbose commentary, the **Agent-Native Cognitive Backlog** is engineered for:

1. **Token Conservation**: Ingesting the entire prioritized portfolio consumes < 100 tokens via a deterministic Tier 1 index.
2. **Zero-Conflict Asynchronous Capture**: Agents (including local small models like `Sanctuary-Qwen2-7B-v1.0` GGUF operating offline) can record unvetted ideas into an append-only JSONL inbox with zero parse ambiguity and zero merge conflicts.
3. **Popperian Empirical Falsification**: Every research mission is treated as an empirical investment requiring an explicit hypothesis, objective falsification criteria, and recorded trial verdicts (`CORROBORATED`, `FALSIFIED`, or `ABANDONED`).
4. **Epistemic Compounding**: Falsifications and corroborations are recorded with citations, updated in manifests, and converted into Layer 3 soul traces (`huggingface/data/soul_traces.jsonl`) for fine-tuning.

---

## 2. Directory Layout & Artifacts

```
LEARNING/research/
├── index.jsonl              # Tier 1 fast-boot machine index (<15 tokens/item, sorted by score)
├── inbox.jsonl              # Append-only intake queue for unvetted agent/human ideas
├── missions/                # Tier 2 individual JSON manifests (single source of truth)
│   ├── MISSION-001.json
│   ├── MISSION-002.json
│   └── ...
└── backlog.md               # Generated human projection (never hand-edited)
```

---

## 3. Mission Lifecycle & State Machine

```
INBOX (inbox.jsonl)
  │ (promote)
  ▼
PROPOSED (missions/*.json) ──▶ STAGED ──▶ ACTIVE ──▶ RESOLVED ──▶ ARCHIVED
  │                           (human      (executing; (verdict:
  │                            gate)       mirror to   CORROBORATED /
  │                                        GH Issue    FALSIFIED)
  ▼                                        if cloud)
PARKED (decay > 30d or blocked)
```

### Transition Invariants:
1. **CAPTURED**: Raw idea captured into `inbox.jsonl` with minimal friction.
2. **PROPOSED**: Promoted into a structured manifest. Score computed via Priority Formula:
   $$\text{Priority Score} = \frac{\text{Strategic Leverage (1–5)} \times \text{Epistemic Depth (1–5)}}{\text{Cost Weight (0.05–4.5)} \times \text{Operational Friction (1–3)}}$$
3. **STAGED**: Human partner issues explicit authorization ("Proceed", "Go", "Execute").
4. **ACTIVE**: Work underway. If `execution_surface == "cloud-worker"`, an advisory GitHub Issue is opened with label `mission:MISSION-NNN`.
5. **RESOLVED**: Empirical trial concluded. Verdict recorded with citations and lessons learned.
6. **ARCHIVED**: Lessons distilled into concepts; trace sealed into Layer 3 JSONL.

---

## 4. Single-Writer Authority & One-Directional Cloud Projection

- **Git-Tracked Local Manifests** are the sole authoritative writable store.
- **GitHub Issues** are purely advisory outward mirrors for remote cloud workers (e.g. Grok Bot on a cloud VM). Cloud workers return results via Git PR modifying the mission manifest's `evidence` field, not by posting unstructured issue comments.
- **Human Markdown View (`backlog.md`)** is strictly a generated read-only projection compiled by `manage_backlog.py compile`. Hand-editing is prohibited.

---

## 4a. Creating a Mission (required checklist)

Whenever an agent drafts a new mission (including replenishment missions), do these in order:

1. **Allocate the ID from `origin/main`**: fetch first; other agents allocate concurrently.
2. **Related-mission scan (required)**: run `manage_backlog.py related --text "<draft title + hypothesis>" --tags <tags>` (or `--id MISSION-NNN` once the file exists) and read the top candidates. The helper is lexical, so it finds topical overlap (shared terms such as routing, token, quota) but misses mechanism-level relations (a mission that attaches to the same lease, ledger or hook). Also ask: *which existing system or mission does this plug into, feed, or depend on?* and check those by hand.
3. **Link bidirectionally**: for each real relation add a `related_missions` entry in the new manifest AND in the related manifest, stating the relationship (DEPENDS ON, CONSUMER, PROVIDER, EXTENDS, VALIDATES, SUPPLIES DATA TO) and whether it is a hard dependency. Declare the edited related manifests in the new mission's `projected_impact`. Do not link on keyword noise alone; a link is a judgement.
4. **Declare `projected_impact`** (paths the mission will touch) so the flight board can lease and detect overlap.
5. **Score** (leverage, depth, cost, friction), run `validate` (structure from `LEARNING/missions/mission.schema.json`, then score and `projected_impact` invariants) and `compile`, and commit the manifest plus regenerated `index.jsonl` and `backlog.md` together.
6. **Status discipline**: a drafted mission stays `PROPOSED`. A flight-board checkout marks the worktree's manifest `IN_PROGRESS`; if you only drafted or researched, restore `PROPOSED` before committing (see MISSION-035's PR #299, which shipped the marker by mistake).

## 5. CLI Tooling

```bash
# Fast Tier 1 inspection
python3 manage_backlog.py list --format jsonl

# Table inspection
python3 manage_backlog.py list

# Suggest related missions before linking (lexical; judge each)
python3 manage_backlog.py related --text "draft title and hypothesis" --tags routing tokens
python3 manage_backlog.py related --id MISSION-035 --top 10

# Frictionless capture from any session/agent
python3 manage_backlog.py capture --idea "Add SHA-256 integrity check to GGUF model downloader" --author qwen

# Record empirical outcome
python3 manage_backlog.py record-result --id MISSION-002 \
  --verdict CORROBORATED \
  --evidence "Fallback chain verified across 71 tests" \
  --citation "PR #744" \
  --lessons "Direct CLI rotation avoids OAuth reverse-proxy failure"

# Recompile index and human markdown view
python3 manage_backlog.py compile

# Invariant validation
python3 manage_backlog.py validate
```

## Manifest schema

`LEARNING/missions/mission.schema.json` (JSON Schema, draft 2020-12 subset) describes one mission manifest and sits beside the
data, so any agent or bundle reading the folder sees it. `scripts/validate_missions.py` is its zero-dependency validator
(`python3 validate_missions.py [--missions-dir DIR] [--schema FILE]`; exit 1 on any error, and an empty folder is an error, never
`OK`). `manage_backlog.py validate` applies it automatically when the root has a schema, and the session wakeup (`guardian_wakeup.py`) reports one `PASS`/`WARN` line for it.

- Unknown optional keys are allowed (drafts must not break); required keys, enums (status, verdict, execution surface, routing tier),
  ID/key patterns, score ranges and types are enforced. File name must equal the manifest `id`.
- `verdict: CONFIRMED` is a LEGACY value (6 resolved missions); `record-result` writes CORROBORATED / FALSIFIED / ABANDONED.
- To change a rule, edit the schema and add a case to `tests/test_validate_missions*.py`; the schema must keep validating every
  existing manifest.
- Origin: written by Claude Haiku 5.5 through the cli-agent skill from field statistics, then integrated and fixed (see
  `LEARNING/experiments/mission-schema/README.md`).


Completion wiring: `record-result` durably marks the final manifest for preservation; after publishing sources to `origin/main`, `merge_prep.py regenerate` runs incremental local extraction, schema and coverage checks. Guardian startup retries pending capture; resume with `soul_coverage.py closure`. Pending/failed/local-complete status is separate from the scientific verdict. Commit the export separately, then sync remotely only with authorization. See the preserving-soul-traces `references/coverage.md` contract.
