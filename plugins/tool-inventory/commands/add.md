---
description: Register a new tool in the inventory (auto-extracts docstring, triggers ChromaDB upsert)
argument-hint: "--path <tool_path> [--category name] [--desc description]"
---

# Add Tool

Register a tool in the JSON inventory. Auto-extracts docstring for description,
detects header compliance style, and upserts the summary into ChromaDB.

## Usage
```bash
# Auto-detect category and description
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py add --path tools/new_script.py

# Explicit category and description
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py add --path tools/cli.py --category orchestrator --desc "CLI router"
```
