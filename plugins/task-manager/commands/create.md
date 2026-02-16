---
description: Create a new task on the kanban board
argument-hint: "\"Task title\" [--objective text] [--status todo|backlog] [--deliverables item1 item2]"
---

# Create Task

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/task_manager.py create "Fix login bug" --objective "Resolve 401 errors" --deliverables "patch auth.py" "add tests"
```
