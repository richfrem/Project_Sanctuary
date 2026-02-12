---
description: Manage Maintenance Tasks (Kanban)
---
# Workflow: Task

1. **List Active Tasks**:
   // turbo
   python3 tools/cli.py task list --status in-progress

2. **Action**:
   - To create: `python3 tools/cli.py task create "Title" --objective "..." --deliverables item1 item2 --acceptance-criteria done1 done2`
   - To update status: `python3 tools/cli.py task update-status N new_status --notes "reason"`
   - To view: `python3 tools/cli.py task get N`
   - To list by status: `python3 tools/cli.py task list --status backlog|todo|in-progress|done`
