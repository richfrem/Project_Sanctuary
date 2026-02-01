---
description: Protocol 128 Phase V - The Technical Seal (Snapshot & Validation)
---
# Workflow: Seal

1. **Gate Check**:
   Confirm you have received Human Approval (Gate 2) from the Audit phase.

2. **Execute Seal**:
   // turbo
   python3 scripts/cortex_cli.py snapshot --type seal

3. **Verify Success**:
   If the command succeeded, proceed to `/workflow-persist`.
   If it failed (Iron Check), you must Backtrack (Recursive Learning).
