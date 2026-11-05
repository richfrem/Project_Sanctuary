# Living Chronicle - Entry 362

**Title:** MISSION-034 corroborated: merge_prep regenerates shared generated files, union-merges the logs, and reports the orphaned antigravity ledger score
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

Plain git merge conflicted 60/60 when two branches both rewrote index.jsonl, backlog.md, the cognitive ledger, events.jsonl and leases.jsonl. With merge=union on the two append-only logs plus merge_prep.py regenerate, 0 of 60 needed a human, 0 differed from a from-scratch regeneration, max 0.026 s per merge, 0 mission manifests touched. Git still marks the three regenerated files conflicted until they are re-added. The tool also found a real defect: cognitive_ledger.json says 16 assessments / 224 points, events.jsonl supports 15 / 209. The extra is one antigravity assessment (15 points, MISSION-024) that exists only in the aggregate; its dimensions and notes are not in git. merge_prep reports it (exit 3) and does not modify the file; agy or a human must log or retract it. Also found: chronicle/ledger.json is stale (335 entries, last id 342 vs 361 files) and nothing maintains it. Measured as a MISSION-035 task: claude-code, task class code, forced past an overlapping own 035 lease (disclosed). Files: scripts/merge_prep.py, tests/test_merge_prep.py, references/merge-prep.md, experiments/mission-034/.
