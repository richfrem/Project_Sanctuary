---
description: Manage Maintenance Tasks (Kanban)
---
# Workflow: Task

1. **List Active Tasks**:
   // turbo
   python3 scripts/domain_cli.py task list --status active

2. **Action**:
   - To create: Use `python3 scripts/domain_cli.py task create "Title" ...`
   - To update: Use `python3 scripts/domain_cli.py task update --id ID ...`
