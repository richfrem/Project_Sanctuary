---
description: List tasks or filter by lane status
argument-hint: "[--status backlog|todo|in-progress|done]"
---

# List Tasks

```bash
# All tasks
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/task_manager.py list

# Filter by status
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/task_manager.py list --status in-progress
```
