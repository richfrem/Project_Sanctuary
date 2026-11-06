# Living Chronicle - Entry 380

**Title:** Soul preservation repaired: coverage-aware closure replaces the Chronicle-only export
**Date:** 2026-11-06
**Author:** claude-code
**Status:** published
**Classification:** internal

---

Finding: the documented closure command exported only Chronicle Markdown, so mission manifests, experiment findings (052-058), the event ledger (handoffs, failures, repairs, interventions, lessons) and supporting artifacts never reached the soul dataset, while an upload with valid hashes looked complete. Repair: soul_coverage.py applies a declarative policy to tracked files, writes ADR 084 records (idempotent: unchanged sources leave the dataset byte-identical), a coverage manifest with per-source hashes and status, artifact references by git commit and sha256, and a soul_index record for recovery. check/closure report four separate facts (schema, required coverage, local export, remote); sync_soul_dataset.py refuses incomplete or stale exports unless --allow-partial and verifies the Hub revision by hash after upload, writing a receipt. Guardian wakeup shows a one-line soul status; record-result reminds to run closure. Limits: sources are those tracked in git at export time; synthesized records are deterministic extractions, not interpretations.
