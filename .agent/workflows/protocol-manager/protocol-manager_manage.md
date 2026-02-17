---
description: List, view, search, or update Protocol documents
argument-hint: "list [--limit N] | get <N> | search \"query\" | update <N> --status STATUS"
---

# Manage Protocols

```bash
# List recent
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/protocol_manager.py list --limit 10

# Filter by status
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/protocol_manager.py list --status CANONICAL

# View specific protocol
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/protocol_manager.py get 128

# Search by keyword
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/protocol_manager.py search "learning loop"

# Update status
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/protocol_manager.py update 128 \
  --status CANONICAL --reason "Approved by council"
```

## Status Values
- 🟡 `PROPOSED` — Draft, under review
- 🟢 `CANONICAL` — Active, enforced
- 🔴 `DEPRECATED` — Superseded or retired
