---
description: Import tools from existing rlm_tool_cache.json into ChromaDB or sync inventory with cache
argument-hint: "[import-json <path>] | [sync-from-cache] | [reset-from-cache]"
---

# Sync / Import Tools

Migrate or sync tool data between JSON cache and ChromaDB.

## Usage
```bash
# Import from existing rlm_tool_cache.json into ChromaDB
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/tool_chroma.py import-json .agent/learning/rlm_tool_cache.json

# Sync descriptions from RLM cache to inventory JSON
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py sync-from-cache

# Full reset: Clear → Discover → Sync → Generate
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py reset-from-cache
```

## Migration Path
When transitioning from `rlm_tool_cache.json` to ChromaDB:
1. Run `import-json` to seed ChromaDB with existing summaries
2. Going forward, `add` and `remove` automatically keep ChromaDB in sync
