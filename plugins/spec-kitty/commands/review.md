---
description: Submit work package for review and move to for_review lane
argument-hint: "<WP-ID>"
---

# Review Work Package

## Step 1: Verify clean state
```bash
python3 tools/orchestrator/verify_workflow_state.py --wp <WP-ID> --phase review
```

## Step 2: Move to for_review
```bash
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE> <WP-ID> for_review \
  --agent "<AGENT-NAME>" --note "Implementation complete, ready for review"
```

## Step 3: Run review
```bash
spec-kitty review <WP-ID>
```
