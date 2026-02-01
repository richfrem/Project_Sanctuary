---
description: Manage Architecture Decision Records (ADR)
---
# Workflow: ADR

1. **List Recent ADRs**:
   // turbo
   python3 scripts/domain_cli.py adr list --limit 5

2. **Action**:
   - To create: Use `/codify-adr` (which calls the template workflow) OR `python3 scripts/domain_cli.py adr create ...`
