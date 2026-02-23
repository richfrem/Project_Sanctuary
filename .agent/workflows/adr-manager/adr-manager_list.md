---
description: List, view, or search Architecture Decision Records
argument-hint: "list [--limit N] | get <number> | search \"query\""
---

# List / Get / Search ADRs

```bash
# List recent
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/adr_manager.py list --limit 5

# View specific ADR
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/adr_manager.py get 35

# Search by keyword
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/adr_manager.py search "chromadb"
```
