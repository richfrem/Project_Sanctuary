---
description: Manage Chronicle Entries (Journaling)
---
# Workflow: Chronicle

1. **List Recent Entries**:
   // turbo
   python3 scripts/domain_cli.py chronicle list --limit 5

2. **Action**:
   - To create: `python3 scripts/domain_cli.py chronicle create "Your Content"`
   - To update: `python3 scripts/domain_cli.py chronicle update --id ID --content "..."`
