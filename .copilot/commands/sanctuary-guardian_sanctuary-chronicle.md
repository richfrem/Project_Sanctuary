---
description: Manage Chronicle Entries (Journaling)
---
# Workflow: Chronicle

1. **List Recent Entries**:
   // turbo
   python3 plugins/chronicle-manager/skills/chronicle-agent/scripts/chronicle_manager.py list --limit 5

2. **Action**:
   - To create: `python3 plugins/chronicle-manager/skills/chronicle-agent/scripts/chronicle_manager.py create "Title" --content "Your Content"`
   - To search: `python3 plugins/chronicle-manager/skills/chronicle-agent/scripts/chronicle_manager.py search "query"`
   - To view: `python3 plugins/chronicle-manager/skills/chronicle-agent/scripts/chronicle_manager.py get N`
