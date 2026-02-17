---
description: Create a new chronicle entry
argument-hint: "\"Title\" --content \"...\" [--author \"Name\"]"
---

# Create Chronicle Entry

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/chronicle_manager.py create "Session Breakthrough" \
  --content "Today the agent achieved..." \
  --author "Guardian" \
  --status draft \
  --classification internal
```

Auto-assigns next 3-digit entry number. Stored in `02_LIVING_CHRONICLE/`.
