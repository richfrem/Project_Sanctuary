---
description: Manage Protocol Documents
---
# Workflow: Protocol

1. **List Recent Protocols**:
   // turbo
   python3 tools/cli.py protocol list --limit 10

2. **Action**:
   - To create: `python3 tools/cli.py protocol create "Title" --content "Protocol content" --status PROPOSED`
   - To search: `python3 tools/cli.py protocol search "query"`
   - To view: `python3 tools/cli.py protocol get N`
   - To update: `python3 tools/cli.py protocol update N --status ACTIVE --reason "Approved by council"`
