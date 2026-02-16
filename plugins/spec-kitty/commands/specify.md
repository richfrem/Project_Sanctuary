---
description: Create or update feature specification from natural language
argument-hint: "\"Feature description\""
---

# Specify Feature

Create the `spec.md` artifact — the **What** and **Why** of a feature.

## Usage
```bash
spec-kitty specify
```

## Verification
```bash
python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase specify
```

**STOP**: Do NOT proceed to Plan until verification passes.
