---
description: Generate work packages (WPs) with subtasks and prompt files
---

# Generate Tasks

Create `tasks.md` and `tasks/WP-*.md` prompt files.

## Usage
```bash
spec-kitty tasks
```

## Verification
```bash
python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase tasks
```

**STOP**: Do NOT proceed to Implementation until verification passes.
