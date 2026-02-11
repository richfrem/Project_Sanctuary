---
description: Protocol 128 Phase IV - Red Team Audit (Capture Learning Snapshot)
---
# Workflow: Audit

> **CLI Command**: `python3 tools/cli.py snapshot --type learning_audit`
> **Output**: `.agent/learning/learning_audit/learning_audit_packet.md`

## Steps

1. **Capture Learning Audit Snapshot**:
   // turbo
   python3 tools/cli.py snapshot --type learning_audit

2. **Wait for Human Review**:
   The snapshot has been generated. Ask the user (Human Gate) to review the
   learning audit packet before proceeding to Seal.

## Snapshot Types Reference

| Type | Purpose | Output |
|------|---------|--------|
| `learning_audit` | Bundle learning materials for red team | `learning_audit_packet.md` |
| `seal` | Final seal before persistence | `learning_package_snapshot.md` |
| `guardian` | Security verification | Guardian report |
| `bootstrap` | Initial project setup | Bootstrap package |

