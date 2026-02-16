---
description: Show kanban board — work package progress across lanes
---

# Status Board

Display kanban board showing WP progress.

## Usage
```bash
# Kanban board via tasks CLI
python3 .kittify/scripts/tasks/tasks_cli.py list <FEATURE-SLUG>

# Or check feature acceptance readiness
python3 .kittify/scripts/tasks/tasks_cli.py status --feature <FEATURE-SLUG>
```

## Lanes
`planned` → `doing` → `for_review` → `done`
