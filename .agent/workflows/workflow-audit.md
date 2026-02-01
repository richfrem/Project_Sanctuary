---
description: Protocol 128 Phase IV - Red Team Audit (Capture Snapshot)
---
# Workflow: Audit

1. **Capture Learning Audit Snapshot**:
   // turbo
   python3 scripts/cortex_cli.py snapshot --type learning_audit

2. **Wait for Human Review**:
   The snapshot has been generated. Ask the user (Human Gate) to review the generic audit packet (or learning packet) before proceeding to Seal.
