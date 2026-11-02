---
name: preserving-soul-traces
description: "Extracts, validates, and seals cognitive session traces into JSONL datasets and lineage manifests for Hugging Face persistence via managing-huggingface-assets. Use when preserving learning as jsonl, exporting soul traces, syncing learning datasets to hugging face, sealing cognitive sessions, or verifying soul trace schemas."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Preserving Soul Traces (`preserving-soul-traces`)

Coordinates the cognitive lineage preservation lifecycle: scanning markdown knowledge, computing SHA-256 integrity digests, formulating canonical JSONL records per ADR 084, maintaining dataset manifests, and delegating remote persistence to the global `managing-huggingface-assets` skill.

## Contents
- [Constraints](#constraints) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints

1. **Human Gate Law**: Uploading datasets to Hugging Face or mutating remote repository state requires explicit human approval ("Proceed", "Go", "Execute").
2. **Zero Code Duplication**: Do NOT implement custom Hugging Face API or OAuth transport; all remote sync must delegate directly to global `managing-huggingface-assets`.
3. **Cryptographic Integrity**: Every JSONL record must compute and verify its SHA-256 hash against sanitized content before persistence.
4. **Binary Stripping**: Image embeds (`![[...]]` and `![alt](...)`) must be stripped to prevent corrupting text token distributions.

## Quick Start

```bash
# 1. Preview extraction from canonical domains without modifying disk
python3 scripts/trace_extractor.py --dry-run

# 2. Extract or update records in the local Hugging Face dataset repo
python3 scripts/trace_extractor.py --append

# 3. Verify JSONL schema compliance and SHA-256 integrity
python3 scripts/trace_extractor.py --verify

# 4. Check readiness of global managing-huggingface-assets integration
python3 scripts/sync_soul_dataset.py --check-readiness

# 5. Dry-run synchronization to richfrem/Project_Sanctuary_Soul
python3 scripts/sync_soul_dataset.py --dry-run
```

## Workflow

1. **Extraction**: Identify new or updated session notes, chronicle entries, or wiki concepts using `trace_extractor.py`.
2. **Normalization & Hashing**: Parse frontmatter metadata and generate SHA-256 digests following [soul-trace-schema.md](references/soul-trace-schema.md).
3. **Manifest Ledger**: Update `hugging_face_dataset_repo/metadata/manifest.json` with updated snapshot counts and dataset hash.
4. **Integrity Audit**: Run `--verify` to ensure zero schema drift or hash corruption.
5. **Remote Persistence**: Delegate folder synchronization to `managing-huggingface-assets` following [huggingface-dataset-integration.md](references/huggingface-dataset-integration.md).

## Verification

```bash
python3 scripts/trace_extractor.py --verify
python3 scripts/sync_soul_dataset.py --check-readiness
```

Review [acceptance-criteria.md](references/acceptance-criteria.md) before declaring dataset sync complete.

## References

- [soul-trace-schema.md](references/soul-trace-schema.md) — 13-field canonical record schema, typing, and validation rules per ADR 084.
- [huggingface-dataset-integration.md](references/huggingface-dataset-integration.md) — Architecture and boundary rules for delegating sync to `managing-huggingface-assets`.
- [acceptance-criteria.md](references/acceptance-criteria.md) — Pre-completion verification checklist.
