---
description: Manage Chronicle Entries (Journaling)
---
# Workflow: Chronicle

1. **List Recent Entries**:
   // turbo
   python3 tools/cli.py chronicle list --limit 5

2. **Action**:
   - To create: `python3 tools/cli.py chronicle create "Title" --content "Your Content"`
   - To search: `python3 tools/cli.py chronicle search "query"`
   - To view: `python3 tools/cli.py chronicle get N`
