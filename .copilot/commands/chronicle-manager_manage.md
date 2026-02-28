---
description: List, view, or search Chronicle entries
argument-hint: "list [--limit N] | get <N> | search \"query\""
---

# Manage Chronicle

```bash
# List recent entries
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/chronicle_manager.py list --limit 5

# View specific entry
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/chronicle_manager.py get 42

# Search by keyword
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/chronicle_manager.py search "learning loop"
```

## Status Values
- 📝 `draft` — Work in progress
- 📗 `published` — Finalized
- 🏛️ `canonical` — Reference-quality
- 🔴 `deprecated` — Superseded

## Classification
- `public` / `internal` / `confidential`
