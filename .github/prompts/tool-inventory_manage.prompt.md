---
description: Full tool update workflow — register, distill, generate docs, audit, verify
argument-hint: "--path <tool_path>"
---

# Manage Tool (Full Workflow)

End-to-end workflow for registering a new or modified tool in the discovery system.

## Steps

### Step 1: Register Tool in Inventory
```bash
# turbo
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py add --path "[ToolPath]"
```

### Step 2: Update Cache (Distillation)
The `add` command auto-triggers ChromaDB upsert. To manually distill via Ollama:
```bash
# turbo
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/distiller.py --file "[ToolPath]" --type tool
```

### Step 3: Generate Markdown Inventory
```bash
# turbo
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py generate --output tools/TOOL_INVENTORY.md
```

### Step 4: Audit for Untracked Tools
```bash
# turbo
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py audit
```

### Step 5: Verify Discovery
```bash
# turbo
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/tool_chroma.py search "[keyword]"
```

## Artifacts Updated

| Artifact | Path | Purpose |
|:---|:---|:---|
| Master Inventory | `tools/tool_inventory.json` | Primary tool registry |
| ChromaDB | `plugins/tool-inventory/data/chroma/` | Semantic search |
| Markdown Inventory | `tools/TOOL_INVENTORY.md` | Human-readable docs |
| JSON Cache (compat) | `.agent/learning/rlm_tool_cache.json` | Legacy search index |
