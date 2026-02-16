---
description: Audit inventory — find missing files, untracked scripts, and ChromaDB coverage gaps
argument-hint: "[--inventory path]"
---

# Audit Inventory

Check for missing files (in JSON but not on disk), untracked scripts (on disk but not in JSON),
and ChromaDB coverage gaps.

## Usage
```bash
# Full audit
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py audit

# ChromaDB stats
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/tool_chroma.py stats
```
