---
description: Create a new Protocol document with auto-numbering
argument-hint: "\"Title\" --content \"...\" [--status PROPOSED|CANONICAL|DEPRECATED]"
---

# Create Protocol

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/protocol_manager.py create "Cognitive Continuity" \
  --content "This protocol defines the learning loop for agent sessions..." \
  --status PROPOSED \
  --classification Internal \
  --authority "Project Sanctuary"
```

Auto-assigns next available protocol number. Stored in `01_PROTOCOLS/`.
