---
description: Manage Protocol Documents
---
# Workflow: Protocol

1. **List Recent Protocols**:
   // turbo
   python3 plugins/protocol-manager/skills/protocol-agent/scripts/protocol_manager.py list --limit 10

2. **Action**:
   - To create: `python3 plugins/protocol-manager/skills/protocol-agent/scripts/protocol_manager.py create "Title" --content "Protocol content" --status PROPOSED`
   - To search: `python3 plugins/protocol-manager/skills/protocol-agent/scripts/protocol_manager.py search "query"`
   - To view: `python3 plugins/protocol-manager/skills/protocol-agent/scripts/protocol_manager.py get N`
   - To update: `python3 plugins/protocol-manager/skills/protocol-agent/scripts/protocol_manager.py update N --status ACTIVE --reason "Approved by council"`
