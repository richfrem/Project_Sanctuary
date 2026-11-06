# Soul coverage: what is preserved, how completeness is verified

**Before (verified 2026-11-06):** the documented closure (`trace_extractor.py --source-dir chronicle/entries --append`) exported Chronicle Markdown only; the extractor handled `.md` only and re-timestamped every record on each run. Mission manifests (hypotheses, verdicts, evidence, lessons), the event ledger (handoffs, failures, repairs, interventions, lessons, evaluation/cost events), experiment READMEs/results from missions 052-058, supporting artifacts and the sealed snapshot were not in the dataset. A hash-valid upload therefore did not mean the learning had been captured.

**Now:** `scripts/soul_coverage.py` applies the declarative policy `references/coverage-policy.json` to the tracked files of the repository.

| source category | critical content | treatment |
|---|---|---|
| chronicle entries | episodic record | Markdown record (ADR 084), unchanged ids |
| mission manifests (`LEARNING/missions/MISSION-*.json`) | hypothesis, falsification, verdict, evidence, lessons, limitations, links | deterministic mission record `mission__MISSION-NNN` |
| experiment Markdown/text (`LEARNING/experiments/**`) | protocols, findings, limitations, deviations, handoffs | Markdown records |
| experiment supporting artifacts (code, raw results, datasets) | reproduction material | reference index per experiment folder (`artifacts__*`): path, bytes, sha256, git commit, on-origin flag; recover with `git show <commit>:<path>` |
| topics / concepts / research / philosophy / procedures docs | dossiers, verified procedures | Markdown records; archived dossiers deduplicated by content |
| event ledger (`.agent/learning/events.jsonl`) | handoffs (CHECKIN), verdicts, failures and repairs, interventions, lessons and their use, deliveries, evaluation and cost events, procedure outcomes | verbatim digest records `events__<TYPE>__NNN` |
| curated memory and identity (`context/{soul,user,memory,status}.md`) | identity, preferences, decisions | Markdown records |
| sealed snapshot (`.agent/learning/learning_package_snapshot.md`) | Protocol 128 seal | Markdown record |
| protocols, ADRs, root architecture docs, skill documentation | governance and design | Markdown records |
| excluded with reason | credentials/auth files, transient state, derived projections (ledgers, index, backlog), kernel telemetry bus, code and tests (versioned in git), receipts | listed in the manifest with the reason |

Anything under a watched root that matches no rule is listed as `uncategorized` in the manifest; a required source that fails to parse, is not UTF-8, is too large, or contains a secret-like string is marked `failed` / `excluded_sensitive` and makes coverage INCOMPLETE (a sensitive line inside the event ledger is skipped, counted and flagged).

## Commands
```bash
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/soul_coverage.py closure     # THE closure command (after a mission is merged)
python3 .../soul_coverage.py check        # four separate facts: SCHEMA, COVERAGE, LOCAL EXPORT, REMOTE
python3 .../soul_coverage.py inventory    # per-category table
python3 .../sync_soul_dataset.py --dry-run   # coverage gate + preview; real sync refuses on incomplete/stale unless --allow-partial
```
`closure` is idempotent: unchanged sources leave the dataset byte-identical (no timestamp churn); a changed source updates exactly its records.

## Four facts, never conflated
1. **SCHEMA** - ADR 084 fields and content hashes valid (`trace_extractor.py --verify`), and every record equals what the manifest recorded.
2. **COVERAGE** - required categories fully included; failed/sensitive/missing/unpublished-artifact items listed.
3. **LOCAL EXPORT** - the export is current with the tracked sources (a changed source makes it STALE).
4. **REMOTE** - a receipt (`.agent/learning/soul_sync_receipts.jsonl`) written after upload by re-reading the Hub revision and comparing the dataset and coverage-manifest hashes. Only when all four hold does the tool print `SOUL PERSISTED`.

Recovery: start from the record `soul_index` (missions, verdicts, record IDs, artifact indexes) and `huggingface/metadata/coverage_manifest.json` (every source with hash, status, record IDs).

## Automatic local completion (2026-11-06)

`manage_backlog.py record-result` atomically saves the final scientific verdict and a
`preservation_obligation` in the same mission manifest. The obligation is durable
and travels through git; it is not an assertion that publication succeeded. Existing
resolved missions and missed integrations are also detected from manifest/export
hashes and the existing export-staleness checks.

The wired post-integration path is `merge_prep.py regenerate`. After regenerating
projections it invokes `preservation_workflow.run`: publication eligibility,
incremental coverage-aware extraction, schema validation and coverage/currentness
checks. `soul_coverage.py closure` invokes the same workflow, and Guardian startup
(both the command and SessionStart wakeup hook) detects and retries pending local
capture. `persist_soul.py --full-sync` now points to this closure. Snapshot-only
`persist_soul.py` remains an explicit transport, not a dataset completion path.
Arbitrary external git merges are not intercepted: run `merge_prep.py regenerate`
after integration; otherwise startup and staleness checks detect missed learning.

Sources must match the locally fetched `origin/main`; supporting artifact commits
must be reachable there. Dirty or unmerged source changes stay `pending_publication`.
Publish the source PR and refresh `origin/main`, then resume closure. No automatic
fetch, commit, push, merge or remote upload is performed. Local capture writes the
generated JSONL and manifests, which require a subsequent scoped export commit/PR.
That export-only commit does not change learning sources and does not trigger a
recursive export. Unchanged closure leaves dataset AND coverage-manifest bytes
unchanged; synchronization receipts are excluded and never stale the export.

Operational status is separate in ignored `.agent/coordination/soul_preservation.json`:
`pending`, `pending_publication`, `failed`, or `local_complete`, with final manifest
hashes, honest verdicts, problems and the resumable command. A pending intent is
saved before extraction. Deleting progress or interrupting the process cannot erase
the tracked obligation; startup reconstructs it. A shared OS lock serializes verdicts,
projection compilation and supported exports across worktrees, and releases on
process death. Dataset/coverage writes use atomic replacement. Direct Python calls
to the low-level extractor are not a supported concurrent lifecycle interface.

Local completion never grants remote authority. Preserve explicit standing approval
when supplied; otherwise report **Local preservation complete. Remote synchronization
pending authorization.** Use the existing `sync_soul_dataset.py` only with authority.
Remote status checks bind receipts to BOTH dataset and coverage-manifest hashes.
Only the four-fact checker may print unqualified `SOUL PERSISTED`.
