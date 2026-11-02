# Acceptance Criteria (`preserving-soul-traces`)

## Verification Checklist

1. **Extraction Integrity**:
   - `trace_extractor.py --dry-run` extracts eligible markdown without side effects.
   - Output contains all 13 canonical fields per ADR 084.
   - Embedded binary references are removed.
   - SHA-256 digests accurately match the UTF-8 payload.

2. **Schema Compliance**:
   - `trace_extractor.py --verify` validates the dataset format and fails on corruption or hash drift.

3. **Delegation Compliance**:
   - `sync_soul_dataset.py --check-readiness` discovers `managing-huggingface-assets` in the global environment.
   - No custom Hugging Face HTTP logic or duplicate credential code exists in `sanctuary-core`.
   - `sync_soul_dataset.py --dry-run` validates target paths without executing network mutations.

4. **Human Gate Law**:
   - Actual upload to `richfrem/Project_Sanctuary_Soul` is only executed when the user explicitly instructs "Proceed", "Go", or "Execute".
