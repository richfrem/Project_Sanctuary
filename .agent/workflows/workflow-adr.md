---
description: Manage Architecture Decision Records (ADR)
---
# Workflow: ADR

1. **List Recent ADRs**:
   // turbo
   python3 tools/cli.py adr list --limit 5

2. **Action**:
   - To create: Use `/codify-adr` (which calls the template workflow) OR `python3 tools/cli.py adr create "Title" --context "..." --decision "..." --consequences "..."`
   - To search: `python3 tools/cli.py adr search "query"`
   - To view: `python3 tools/cli.py adr get N`
