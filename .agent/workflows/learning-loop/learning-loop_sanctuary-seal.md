---
description: Protocol 128 Phase VI - The Technical Seal (Snapshot & Validation)
---
# Workflow: Seal

> **CLI Command**: `python3 tools/cli.py snapshot --type seal`
> **Output**: `.agent/learning/learning_package_snapshot.md`

## Steps

1. **Gate Check**:
   Confirm you have received Human Approval (Gate 2) from the Audit phase.

2. **Execute Seal**:
   // turbo
   python3 tools/cli.py snapshot --type seal

3. **Verify Success**:
   If the command succeeded, proceed to `/sanctuary-persist`.
   If it failed (Iron Check), you must Backtrack to Phase VIII (Self-Correction).

